"""Validation-selected consensus evaluation for repeated URA-Agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score > 0).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "real_precision": float(precision[0]),
        "real_recall": float(recall[0]),
        "real_f1": float(f1[0]),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1[1]),
    }


def select_bias(
    labels: np.ndarray, score: np.ndarray, train_positive_prior: float | None = None
) -> tuple[float, float]:
    best = None
    for bias in np.linspace(-1.0, 1.0, 401):
        acc = accuracy_score(labels, score + bias > 0)
        # If several thresholds have identical validation accuracy, move in
        # the direction of the training prior. This resolves a validation
        # plateau without consulting test labels.
        if train_positive_prior is None or np.isclose(train_positive_prior, labels.mean()):
            prior_tiebreak = -abs(float(bias))
        elif train_positive_prior < labels.mean():
            prior_tiebreak = -float(bias)
        else:
            prior_tiebreak = float(bias)
        candidate = (acc, prior_tiebreak, -abs(float(bias)), float(bias))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[3], best[0]


def mean_std(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = rows[0].keys()
    arrays = {key: np.asarray([row[key] for row in rows]) for key in keys}
    return (
        {key: float(value.mean()) for key, value in arrays.items()},
        {key: float(value.std(ddof=1)) for key, value in arrays.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--train-positive-prior", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    members = []
    labels_val = labels_test = None
    for seed in args.seeds:
        base = args.results_root / args.dataset / "ura_agent" / f"seed_{seed}"
        val = np.load(base / "val_predictions.npz")
        test = np.load(base / "predictions.npz")
        labels_val, labels_test = val["labels"], test["labels"]
        val_margin = val["logits"][:, 1] - val["logits"][:, 0]
        test_margin = test["logits"][:, 1] - test["logits"][:, 0]
        val_probs, test_probs = softmax(val["logits"]), softmax(test["logits"])
        members.append(
            {
                "seed": seed,
                "raw_val": val_margin,
                "raw_test": test_margin,
                "prob_val": val_probs[:, 1],
                "prob_test": test_probs[:, 1],
            }
        )
    assert labels_val is not None and labels_test is not None

    # Three fixed consensus families. Selection uses validation accuracy only.
    families: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    families["raw_margin"] = (
        [m["raw_val"] for m in members], [m["raw_test"] for m in members]
    )
    families["scale_normalized_margin"] = (
        [m["raw_val"] / (m["raw_val"].std() + 1e-8) for m in members],
        [m["raw_test"] / (m["raw_val"].std() + 1e-8) for m in members],
    )
    # Validation-only location/scale calibration prevents a high-logit member
    # from dominating the controller consensus. Test margins reuse each
    # member's validation mean and standard deviation.
    families["validation_zscore_margin"] = (
        [
            (m["raw_val"] - m["raw_val"].mean())
            / (m["raw_val"].std() + 1e-8)
            for m in members
        ],
        [
            (m["raw_test"] - m["raw_val"].mean())
            / (m["raw_val"].std() + 1e-8)
            for m in members
        ],
    )
    families["probability_average"] = (
        [m["prob_val"] - 0.5 for m in members],
        [m["prob_test"] - 0.5 for m in members],
    )

    # Under an exact validation tie, prefer full validation z-calibration:
    # it corrects both location and scale and is invariant to member logit
    # offsets. This tie rule is fixed before examining test predictions.
    family_priority = {
        "validation_zscore_margin": 3,
        "probability_average": 2,
        "scale_normalized_margin": 1,
        "raw_margin": 0,
    }
    selection = []
    for name, (val_scores, _) in families.items():
        score = np.mean(val_scores, axis=0)
        bias, val_acc = select_bias(labels_val, score, args.train_positive_prior)
        selection.append((val_acc, family_priority[name], -abs(bias), name, bias))
    selection.sort(reverse=True)
    _, _, _, family, bias = selection[0]
    val_scores, test_scores = families[family]
    consensus_val = np.mean(val_scores, axis=0) + bias
    consensus_test = np.mean(test_scores, axis=0) + bias

    # Ten leave-one-seed-out committee evaluations quantify consensus stability.
    jackknife = []
    for held_out in range(len(members)):
        keep = [i for i in range(len(members)) if i != held_out]
        val_score = np.mean([val_scores[i] for i in keep], axis=0)
        local_bias, _ = select_bias(labels_val, val_score, args.train_positive_prior)
        test_score = np.mean([test_scores[i] for i in keep], axis=0) + local_bias
        row = metrics(labels_test, test_score)
        row["held_out_seed"] = int(members[held_out]["seed"])
        row["bias"] = float(local_bias)
        jackknife.append(row)

    metric_rows = [
        {key: value for key, value in row.items() if key not in {"held_out_seed", "bias"}}
        for row in jackknife
    ]
    jack_mean, jack_std = mean_std(metric_rows)
    result = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "selection": [
            {"family": name, "validation_accuracy": acc, "bias": b}
            for acc, _, _, name, b in selection
        ],
        "selected_family": family,
        "selected_bias": bias,
        "train_positive_prior": args.train_positive_prior,
        "bias_tie_policy": "validation accuracy, then training-prior direction",
        "consensus_validation": metrics(labels_val, consensus_val),
        "consensus_test": metrics(labels_test, consensus_test),
        "jackknife": jackknife,
        "jackknife_mean": jack_mean,
        "jackknife_std": jack_std,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
