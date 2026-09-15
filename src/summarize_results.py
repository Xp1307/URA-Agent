"""Aggregate multi-seed metrics and paired-bootstrap URA-Agent comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


METRICS = (
    "accuracy",
    "macro_f1",
    "fake_precision",
    "fake_recall",
    "fake_f1",
    "real_f1",
    "auroc",
    "ece",
    "avg_optional_calls",
)


def read_runs(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*/seed_*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "dataset": result["config"]["dataset"],
            "method": result["config"]["method"],
            "seed": result["config"]["seed"],
            "best_epoch": result["best_epoch"],
            "runtime_seconds": result["runtime_seconds"],
            "n_trainable": result["n_trainable"],
            "n_total": result["n_total"],
        }
        row.update(result["test"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
    result = []
    for (dataset, method), group in sorted(groups.items()):
        out = {"dataset": dataset, "method": method, "n_seeds": len(group)}
        for metric in METRICS:
            values = np.asarray([r[metric] for r in group if metric in r], dtype=float)
            if len(values):
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(out)
    return result


def paired_bootstrap(
    root: Path,
    dataset: str,
    baseline: str,
    seeds: list[int],
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    pairs = []
    for seed in seeds:
        agent_path = root / dataset / "ura_agent" / f"seed_{seed}" / "predictions.npz"
        base_path = root / dataset / baseline / f"seed_{seed}" / "predictions.npz"
        if not agent_path.exists() or not base_path.exists():
            continue
        agent, base = np.load(agent_path), np.load(base_path)
        if not np.array_equal(agent["ids"], base["ids"]):
            raise ValueError(f"Prediction IDs differ for {dataset}, seed={seed}")
        if not np.array_equal(agent["labels"], base["labels"]):
            raise ValueError(f"Labels differ for {dataset}, seed={seed}")
        pairs.append((agent["labels"], agent["logits"].argmax(1), base["logits"].argmax(1)))
    if not pairs:
        return {}
    def batched_macro_f1(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
        tp = ((labels == 1) & (pred == 1)).sum(1)
        fp = ((labels == 0) & (pred == 1)).sum(1)
        fn = ((labels == 1) & (pred == 0)).sum(1)
        tn = ((labels == 0) & (pred == 0)).sum(1)
        f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
        f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fn + fp) != 0)
        return 0.5 * (f1_pos + f1_neg)

    deltas = np.empty(iterations, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        per_seed = []
        for labels, agent_pred, base_pred in pairs:
            idx = rng.integers(0, len(labels), size=(stop - start, len(labels)))
            sampled_labels = labels[idx]
            per_seed.append(
                batched_macro_f1(sampled_labels, agent_pred[idx])
                - batched_macro_f1(sampled_labels, base_pred[idx])
            )
        deltas[start:stop] = np.mean(per_seed, axis=0)
    observed = np.mean(
        [
            f1_score(y, a, average="macro") - f1_score(y, b, average="macro")
            for y, a, b in pairs
        ]
    )
    return {
        "dataset": dataset,
        "baseline": baseline,
        "n_seeds": len(pairs),
        "macro_f1_delta": float(observed),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/strong_results"))
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = read_runs(args.results_root)
    summary = aggregate(rows)
    write_csv(args.results_root / "strong_seed_results.csv", rows)
    write_csv(args.results_root / "strong_main_results.csv", summary)

    significance = []
    rng = np.random.default_rng(2026)
    for dataset in sorted({r["dataset"] for r in rows}):
        complete = [r for r in summary if r["dataset"] == dataset and r["n_seeds"] >= 3]
        baselines = [r for r in complete if r["method"] != "ura_agent"]
        agent = next((r for r in complete if r["method"] == "ura_agent"), None)
        if not baselines or agent is None:
            continue
        strongest = max(baselines, key=lambda r: r["macro_f1_mean"])["method"]
        seeds = sorted(
            r["seed"] for r in rows if r["dataset"] == dataset and r["method"] == "ura_agent"
        )
        significance.append(
            paired_bootstrap(args.results_root, dataset, strongest, seeds, args.iterations, rng)
        )
    (args.results_root / "significance.json").write_text(
        json.dumps(significance, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "significance": significance}, indent=2))


if __name__ == "__main__":
    main()
