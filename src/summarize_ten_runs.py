"""Summarize ten independent URA-Agent test runs as mean +/- sample std."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


METRIC_KEYS = (
    "accuracy",
    "real_precision",
    "real_recall",
    "real_f1",
    "fake_precision",
    "fake_recall",
    "fake_f1",
)


def calculate(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "real_precision": float(precision[0]),
        "real_recall": float(recall[0]),
        "real_f1": float(f1[0]),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1[1]),
    }


def format_percent(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}±{100.0 * std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", default=["weibo", "pheme"])
    parser.add_argument("--method", default="ura_agent")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary: dict[str, object] = {
        "method": args.method,
        "seeds": args.seeds,
        "std": "sample standard deviation (ddof=1)",
        "label_mapping": {"real": 0, "fake": 1},
        "datasets": {},
    }
    md = [
        "| Dataset | Accuracy | Real P | Real R | Real F1 | Fake P | Fake R | Fake F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in args.datasets:
        rows = []
        for seed in args.seeds:
            path = args.results_root / dataset / args.method / f"seed_{seed}" / "predictions.npz"
            arrays = np.load(path)
            labels = arrays["labels"].astype(np.int64)
            if "predictions" in arrays:
                predictions = arrays["predictions"].astype(np.int64)
            elif "preds" in arrays:
                predictions = arrays["preds"].astype(np.int64)
            else:
                predictions = arrays["logits"].argmax(axis=1).astype(np.int64)
            row = {"seed": seed, **calculate(labels, predictions)}
            rows.append(row)

        mean = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
        std = {
            key: float(np.std([row[key] for row in rows], ddof=1)) for key in METRIC_KEYS
        }
        summary["datasets"][dataset] = {"runs": rows, "mean": mean, "std": std}
        values = [format_percent(mean[key], std[key]) for key in METRIC_KEYS]
        md.append(f"| {dataset.capitalize()} | " + " | ".join(values) + " |")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
