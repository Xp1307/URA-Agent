"""Evaluate URA-Agent component ablations from cached stage predictions.

The script reports both independent per-seed statistics and the fixed-policy
leave-one-member-out committee stability protocol used in the paper analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01)
ABLATIONS = {
    "Text-only": (0,),
    "w/o Vision": (0, 2, 3),
    "w/o Grounding": (0, 1, 3),
    "w/o Consistency": (0, 1, 2),
    "Full URA-Agent": (0, 1, 2, 3),
}


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def route(stages: np.ndarray, order: tuple[int, ...], thresholds: tuple[float, ...]):
    current = stages[:, order[0]].copy()
    calls = np.zeros(len(current), dtype=np.float32)
    if len(order) == 1:
        return current, 0.0
    active = softmax(current).max(1) < thresholds[0]
    for position, expert in enumerate(order[1:], start=1):
        current[active] = stages[active, expert]
        calls[active] += 1
        if position < len(order) - 1:
            active &= softmax(current).max(1) < thresholds[position]
    return current, float(calls.mean())


def metrics(labels: np.ndarray, logits_or_probs: np.ndarray, is_probs: bool = False):
    probs = logits_or_probs if is_probs else softmax(logits_or_probs)
    pred = probs.argmax(1)
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


def score_metrics(labels: np.ndarray, score: np.ndarray):
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


def select_bias(labels: np.ndarray, score: np.ndarray, train_positive_prior: float):
    best = None
    for bias in np.linspace(-1.0, 1.0, 401):
        accuracy = accuracy_score(labels, score + bias > 0)
        if train_positive_prior < labels.mean():
            prior_tiebreak = -float(bias)
        elif train_positive_prior > labels.mean():
            prior_tiebreak = float(bias)
        else:
            prior_tiebreak = -abs(float(bias))
        candidate = (accuracy, prior_tiebreak, -abs(float(bias)), float(bias))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return best[3], best[0]


def consensus_ablation(
    labels_val: np.ndarray,
    labels_test: np.ndarray,
    val_logits: list[np.ndarray],
    test_logits: list[np.ndarray],
    train_positive_prior: float,
):
    raw_val = [x[:, 1] - x[:, 0] for x in val_logits]
    raw_test = [x[:, 1] - x[:, 0] for x in test_logits]
    families = {
        "raw_margin": (raw_val, raw_test),
        "scale_normalized_margin": (
            [x / (x.std() + 1e-8) for x in raw_val],
            [x / (v.std() + 1e-8) for x, v in zip(raw_test, raw_val)],
        ),
        "validation_zscore_margin": (
            [(x - x.mean()) / (x.std() + 1e-8) for x in raw_val],
            [
                (x - v.mean()) / (v.std() + 1e-8)
                for x, v in zip(raw_test, raw_val)
            ],
        ),
        "probability_average": (
            [softmax(x)[:, 1] - 0.5 for x in val_logits],
            [softmax(x)[:, 1] - 0.5 for x in test_logits],
        ),
    }
    priority = {
        "validation_zscore_margin": 3,
        "probability_average": 2,
        "scale_normalized_margin": 1,
        "raw_margin": 0,
    }
    candidates = []
    for name, (val_scores, _) in families.items():
        score = np.mean(val_scores, axis=0)
        bias, accuracy = select_bias(labels_val, score, train_positive_prior)
        candidates.append((accuracy, priority[name], name, bias))
    candidates.sort(reverse=True)
    _, _, family, bias = candidates[0]
    val_scores, test_scores = families[family]
    full_score = np.mean(test_scores, axis=0) + bias
    full = score_metrics(labels_test, full_score)

    jackknife = []
    fixed_policy_runs = []
    for held_out in range(len(test_scores)):
        keep = [i for i in range(len(test_scores)) if i != held_out]
        val_score = np.mean([val_scores[i] for i in keep], axis=0)
        local_bias, _ = select_bias(
            labels_val, val_score, train_positive_prior
        )
        test_score = np.mean([test_scores[i] for i in keep], axis=0) + local_bias
        result = score_metrics(labels_test, test_score)
        result["held_out_index"] = held_out
        result["bias"] = local_bias
        jackknife.append(result)
        fixed_result = score_metrics(
            labels_test,
            np.mean([test_scores[i] for i in keep], axis=0) + bias,
        )
        fixed_result["held_out_index"] = held_out
        fixed_result["bias"] = bias
        fixed_policy_runs.append(fixed_result)
    keys = tuple(full.keys())
    mean = {key: float(np.mean([x[key] for x in jackknife])) for key in keys}
    std = {key: float(np.std([x[key] for x in jackknife], ddof=1)) for key in keys}
    fixed_mean = {
        key: float(np.mean([x[key] for x in fixed_policy_runs])) for key in keys
    }
    fixed_std = {
        key: float(np.std([x[key] for x in fixed_policy_runs], ddof=1)) for key in keys
    }
    return {
        "selected_family": family,
        "selected_bias": bias,
        "family_selection": [
            {"family": name, "validation_accuracy": accuracy, "bias": local_bias}
            for accuracy, _, name, local_bias in candidates
        ],
        "full_committee": full,
        "jackknife": jackknife,
        "jackknife_mean": mean,
        "jackknife_std": std,
        "fixed_policy_member_perturbation": fixed_policy_runs,
        "fixed_policy_mean": fixed_mean,
        "fixed_policy_std": fixed_std,
    }


def tune(labels: np.ndarray, stages: np.ndarray, order: tuple[int, ...]):
    if len(order) == 1:
        return (), metrics(labels, stages[:, order[0]])
    best = None
    for thresholds in itertools.product(GRID, repeat=len(order) - 1):
        logits, calls = route(stages, order, thresholds)
        result = metrics(labels, logits)
        candidate = (result["accuracy"], -calls, thresholds, result)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    best[3]["calls"] = -best[1]
    return best[2], best[3]


def pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def markdown_table(rows: list[dict]) -> str:
    header = (
        "| Ablation | Accuracy | Real P | Real R | Real F1 | "
        "Fake P | Fake R | Fake F1 | Calls |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {name} | {accuracy} | {real_precision} | "
            "{real_recall} | {real_f1} | {fake_precision} | "
            "{fake_recall} | {fake_f1} | {calls} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[13, 21, 42, 55, 67, 73, 87, 101, 123, 2026],
    )
    parser.add_argument("--datasets", nargs="+", default=["weibo", "pheme"])
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/strong_results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    roots = {dataset: args.project_root / args.results_root for dataset in args.datasets}
    for dataset, root in roots.items():
        routed_by_name: dict[str, list[np.ndarray]] = {name: [] for name in ABLATIONS}
        routed_val_logits: dict[str, list[np.ndarray]] = {name: [] for name in ABLATIONS}
        routed_test_logits: dict[str, list[np.ndarray]] = {name: [] for name in ABLATIONS}
        labels_val = labels_test = None
        seed_results = {name: [] for name in ABLATIONS}
        call_results = {name: [] for name in ABLATIONS}
        policies = {name: {} for name in ABLATIONS}

        for seed in args.seeds:
            base = root / dataset / "ura_agent" / f"seed_{seed}"
            val = np.load(base / "val_predictions.npz")
            test = np.load(base / "stage_predictions.npz")
            labels_val = val["labels"]
            labels_test = test["labels"]
            for name, order in ABLATIONS.items():
                thresholds, _ = tune(val["labels"], val["stage_logits"], order)
                val_logits, _ = route(val["stage_logits"], order, thresholds)
                logits, calls = route(test["stage_logits"], order, thresholds)
                routed_val_logits[name].append(val_logits)
                routed_test_logits[name].append(logits)
                routed_by_name[name].append(softmax(logits))
                result = metrics(labels_test, logits)
                result["calls"] = calls
                seed_results[name].append(result)
                call_results[name].append(calls)
                policies[name][str(seed)] = list(thresholds)

        rows = []
        detailed = {}
        keys = list(metrics(np.array([0, 1]), np.array([[1, 0], [0, 1]])).keys())
        for name in ABLATIONS:
            ensemble_probs = np.mean(routed_by_name[name], axis=0)
            ensemble = metrics(labels_test, ensemble_probs, is_probs=True)
            ensemble["calls"] = float(np.mean(call_results[name]))
            arrays = {key: np.array([r[key] for r in seed_results[name]]) for key in keys}
            mean = {key: float(value.mean()) for key, value in arrays.items()}
            std = {key: float(value.std(ddof=1)) for key, value in arrays.items()}
            rows.append({
                "name": name,
                **{key: pm(mean[key], std[key]) for key in keys},
                "calls": pm(float(np.mean(call_results[name])), float(np.std(call_results[name], ddof=1)), 3),
            })
            detailed[name] = {
                "order": list(ABLATIONS[name]),
                "policies": policies[name],
                "per_seed": seed_results[name],
                "mean": mean,
                "std": std,
                "ensemble": ensemble,
            }

        # No-routing baseline: uniformly average all expert probabilities.
        uniform_seed_probs, uniform_seed_results = [], []
        uniform_val_logits, uniform_test_logits = [], []
        for seed in args.seeds:
            base = root / dataset / "ura_agent" / f"seed_{seed}"
            val = np.load(base / "val_predictions.npz")
            test = np.load(base / "stage_predictions.npz")
            val_probs = softmax(val["stage_logits"]).mean(1)
            probs = softmax(test["stage_logits"]).mean(1)
            uniform_val_logits.append(np.log(val_probs.clip(1e-8, 1.0)))
            uniform_test_logits.append(np.log(probs.clip(1e-8, 1.0)))
            uniform_seed_probs.append(probs)
            uniform_seed_results.append(metrics(test["labels"], probs, is_probs=True))
        uniform_ensemble = metrics(labels_test, np.mean(uniform_seed_probs, axis=0), is_probs=True)
        uniform_ensemble["calls"] = 3.0
        uniform_arrays = {
            key: np.array([result[key] for result in uniform_seed_results]) for key in keys
        }
        uniform_mean = {key: float(value.mean()) for key, value in uniform_arrays.items()}
        uniform_std = {key: float(value.std(ddof=1)) for key, value in uniform_arrays.items()}
        rows.insert(-1, {
            "name": "w/o Adaptive Router",
            **{key: pm(uniform_mean[key], uniform_std[key]) for key in keys},
            "calls": "3.000±0.000",
        })
        detailed["w/o Adaptive Router"] = {
            "per_seed": uniform_seed_results,
            "mean": uniform_mean,
            "std": uniform_std,
            "ensemble": uniform_ensemble,
        }

        # Evaluate the final multi-seed decision protocol used by URA-Agent.
        # The final calibration is fixed once using validation predictions.
        # Ten leave-one-member-out perturbations then provide mean +/- sample std.
        train_path = args.project_root / "datasets" / dataset / "data_json" / "train_imdes.json"
        if not train_path.exists():
            train_path = args.project_root / "datasets" / dataset / "data_json" / "train.json"
        train_payload = json.loads(train_path.read_text(encoding="utf-8"))
        train_records = (
            list(train_payload.values()) if isinstance(train_payload, dict) else train_payload
        )
        train_positive_prior = float(
            np.mean([int(record["tweet_label"]) for record in train_records])
        )
        committee_rows = []
        for name in ABLATIONS:
            committee = consensus_ablation(
                labels_val,
                labels_test,
                routed_val_logits[name],
                routed_test_logits[name],
                train_positive_prior,
            )
            detailed[name]["committee"] = committee
            committee_rows.append({
                "name": name,
                **{
                    key: pm(
                        committee["fixed_policy_mean"][key],
                        committee["fixed_policy_std"][key],
                    )
                    for key in keys
                },
                "calls": pm(
                    float(np.mean(call_results[name])),
                    float(np.std(call_results[name], ddof=1)),
                    3,
                ),
            })
        uniform_committee = consensus_ablation(
            labels_val,
            labels_test,
            uniform_val_logits,
            uniform_test_logits,
            train_positive_prior,
        )
        detailed["w/o Adaptive Router"]["committee"] = uniform_committee
        committee_rows.insert(-1, {
            "name": "w/o Adaptive Router",
            **{
                key: pm(
                    uniform_committee["fixed_policy_mean"][key],
                    uniform_committee["fixed_policy_std"][key],
                )
                for key in keys
            },
            "calls": "3.000±0.000",
        })
        committee_table = markdown_table(committee_rows)
        (args.output_dir / f"{dataset}_ablation_committee.md").write_text(
            committee_table + "\n", encoding="utf-8"
        )

        table = markdown_table(rows)
        (args.output_dir / f"{dataset}_ablation.md").write_text(table + "\n", encoding="utf-8")
        (args.output_dir / f"{dataset}_ablation.json").write_text(
            json.dumps(detailed, indent=2), encoding="utf-8"
        )
        all_results[dataset] = {
            "independent_table": table,
            "committee_table": committee_table,
            "details": detailed,
        }
        print(f"\n## {dataset.upper()} - INDEPENDENT\n\n{table}")
        print(f"\n## {dataset.upper()} - FIXED-POLICY MEMBER PERTURBATION\n\n{committee_table}")

    (args.output_dir / "ura_ablation_all.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
