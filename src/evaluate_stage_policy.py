"""Select one conservative multi-agent residual policy on validation predictions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def class_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = (scores > 0).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(
        labels, pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "real_precision": float(p[0]),
        "real_recall": float(r[0]),
        "real_f1": float(f[0]),
        "fake_precision": float(p[1]),
        "fake_recall": float(r[1]),
        "fake_f1": float(f[1]),
    }


def load_stage(root: Path, dataset: str, seed: int, split: str):
    base = root / dataset / "ura_agent" / f"seed_{seed}"
    if split == "val":
        a = np.load(base / "val_predictions.npz")
        stage = a["stage_logits"]
    else:
        a = np.load(base / "stage_predictions.npz")
        stage = a["stage_logits"]
    margin = stage[:, :, 1] - stage[:, :, 0]
    return a["labels"].astype(np.int64), margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    members = []
    for seed in args.seeds:
        yv, mv = load_stage(args.results_root, args.dataset, seed, "val")
        yt, mt = load_stage(args.results_root, args.dataset, seed, "test")
        # Calibrate each specialist exclusively with validation statistics.
        center = mv.mean(axis=0, keepdims=True)
        scale = mv.std(axis=0, keepdims=True) + 1e-8
        members.append(
            {
                "seed": seed,
                "yv": yv,
                "yt": yt,
                "zv": (mv - center) / scale,
                "zt": (mt - center) / scale,
            }
        )

    # Text is the anchor (weight 1). Auxiliary specialists can only make a
    # conservative residual correction. One policy is shared across all seeds.
    residual_grid = [0.0, 0.15, 0.30, 0.45, 0.60]
    bias_grid = np.linspace(-0.40, 0.40, 161)
    best = None
    for residuals in itertools.product(residual_grid, repeat=3):
        weights = np.asarray((1.0, *residuals))
        val_base = [m["zv"] @ weights for m in members]
        for bias in bias_grid:
            accs = [accuracy_score(m["yv"], score + bias > 0) for m, score in zip(members, val_base)]
            # First maximize average validation accuracy; then minimize
            # instability and policy complexity to reduce small-val overfit.
            candidate = (
                float(np.mean(accs)),
                -float(np.std(accs)),
                -float(sum(residuals)),
                -abs(float(bias)),
                tuple(float(x) for x in residuals),
                float(bias),
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    assert best is not None
    residuals, bias = best[4], best[5]
    weights = np.asarray((1.0, *residuals))

    rows = []
    for member in members:
        val_score = member["zv"] @ weights + bias
        test_score = member["zt"] @ weights + bias
        rows.append(
            {
                "seed": member["seed"],
                "validation": class_metrics(member["yv"], val_score),
                "test": class_metrics(member["yt"], test_score),
            }
        )
    keys = rows[0]["test"].keys()
    mean = {k: float(np.mean([r["test"][k] for r in rows])) for k in keys}
    std = {k: float(np.std([r["test"][k] for r in rows], ddof=1)) for k in keys}
    result = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "selection_data": "validation only",
        "stage_order": ["text", "image", "grounding", "consistency"],
        "weights": weights.tolist(),
        "bias": bias,
        "mean_validation_accuracy": best[0],
        "runs": rows,
        "test_mean": mean,
        "test_std": std,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
