"""Audit the stopping depth of fixed URA-Agent models; never tune on test data.

Replays cached per-agent logits with the already saved hard-routing thresholds.
Checks every routed prediction against both the original routing function and
the saved final prediction. No training or backbone inference is performed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


AGENTS = ("Text", "Vision", "Grounding", "Consistency")
SEEDS = (13, 21, 42, 55, 67, 73, 87, 101, 123, 2026)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_router(path: Path):
    """Load only the pure original route_agent function, not model imports."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "route_agent"
    )
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["route_agent"]


def trace_route(stage_logits: np.ndarray, thresholds: list[float]):
    """Absorbing STOP: a stopped sample can never be activated again."""
    stages = torch.from_numpy(stage_logits)
    assert stages.ndim == 3 and stages.shape[1:] == (4, 2)
    assert len(thresholds) == 3 and torch.isfinite(stages).all()
    n = len(stages)
    current = stages[:, 0].clone()
    depth = torch.zeros(n, dtype=torch.int64)
    alive = torch.ones(n, dtype=torch.bool)
    reached = torch.zeros((n, 4), dtype=torch.bool)
    reached[:, 0] = True
    for j, threshold in enumerate(thresholds):
        call = alive & (current.softmax(-1).max(-1).values < threshold)
        current[call] = stages[call, j + 1]
        reached[call, j + 1] = True
        depth[call] = j + 1
        alive = call
    conf = current.softmax(-1).max(-1).values.numpy()
    depth = depth.numpy()
    reached = reached.numpy()
    assert np.array_equal(reached, np.arange(4)[None, :] <= depth[:, None])
    return current.numpy(), depth, reached, conf


def test_trace(reference):
    probabilities = np.array([
        [[.95, .05], [.50, .50], [.50, .50], [.50, .50]],
        [[.50, .50], [.95, .05], [.50, .50], [.50, .50]],
        [[.50, .50], [.50, .50], [.95, .05], [.50, .50]],
        [[.50, .50], [.50, .50], [.50, .50], [.80, .20]],
    ], dtype=np.float32)
    logits = np.log(probabilities)
    out, depth, reached, _ = trace_route(logits, [.9, .9, .9])
    assert depth.tolist() == [0, 1, 2, 3]
    assert reached.sum(0).tolist() == [4, 3, 2, 1]
    ref, calls = reference(logits, [.9, .9, .9])
    np.testing.assert_array_equal(out, ref)
    assert float(depth.mean()) == calls
    assert trace_route(logits, [1.01] * 3)[1].tolist() == [3] * 4
    assert trace_route(logits, [.50] * 3)[1].tolist() == [0] * 4
    assert trace_route(np.zeros_like(logits), [.50] * 3)[1].tolist() == [0] * 4
    # Decisions and path depth have no dependency on labels.


def summary_stat(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    print("Starting stopping-depth audit", flush=True)
    torch.set_num_threads(1)
    root = args.project_root.resolve()
    result_root = (root / args.results_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "src/train_strong.py"
    print("Loading original routing function", flush=True)
    reference = original_router(source)
    print("Running synthetic boundary and absorbing-stop tests", flush=True)
    test_trace(reference)
    print("Synthetic tests passed", flush=True)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "results_root": str(result_root),
        "analysis": "Cached-logit replay of fixed per-seed confidence-based hard routing",
        "agent_order": list(AGENTS),
        "seeds": args.seeds,
        "split": "test",
        "threshold_source": "existing metrics.json routing_threshold; no retuning",
        "statistics": "equal-weight seed mean and sample standard deviation (ddof=1)",
        "stopping_definition": "depth==j / N; four mutually exclusive shares sum to 100%",
        "reach_definition": "depth>=j / N; includes samples continuing to later agents",
        "conditional_stop_definition": "stop_count / reach_count; undefined if reach_count==0",
        "last_stage": "Consistency is terminal; no fourth confidence test",
        "labels_used_for": "accuracy validation only, never stop/call decisions",
        "compute_caveat": "logical routing only: existing inference precomputes all experts; not a latency/FLOP measurement",
        "committee_caveat": "per-seed member paths, not one common stopping depth for a multi-model committee",
        "source_sha256": {str(source): sha256(source)},
        "datasets": {},
        "checks": {"unit_tests": "passed", "all_saved_predictions_match": True},
    }
    per_seed_rows, stage_rows, sample_rows = [], [], []
    for dataset in args.datasets:
        runs, seed_depths = [], []
        ref_ids = ref_labels = None
        ablation_path = root / "outputs/ablation" / f"{dataset}_ablation.json"
        ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None
        for seed in args.seeds:
            print(f"Auditing {dataset} seed={seed}", flush=True)
            base = result_root / dataset / "ura_agent" / f"seed_{seed}"
            metrics_path = base / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            threshold = metrics["routing_threshold"]
            threshold = [float(threshold)] * 3 if np.isscalar(threshold) else list(map(float, threshold))
            assert metrics["config"]["selection_metric"] == "accuracy"
            assert metrics["config"]["merge_val"] is False
            if ablation is not None:
                assert threshold == ablation["Full URA-Agent"]["policies"][str(seed)]
            with np.load(base / "stage_predictions.npz", allow_pickle=False) as f:
                ids, labels, stages = f["ids"], f["labels"], f["stage_logits"]
            with np.load(base / "predictions.npz", allow_pickle=False) as f:
                saved_ids, saved_labels, saved_logits = f["ids"], f["labels"], f["logits"]
            assert np.array_equal(ids, saved_ids) and np.array_equal(labels, saved_labels)
            assert len(ids) == len(set(map(str, ids))), "Duplicate sample IDs require review"
            if ref_ids is None:
                ref_ids, ref_labels = ids, labels
            else:
                assert np.array_equal(ids, ref_ids) and np.array_equal(labels, ref_labels)
            routed, depth, reached, confidence = trace_route(stages, threshold)
            reference_logits, reference_calls = reference(stages, threshold)
            np.testing.assert_array_equal(routed, reference_logits)
            np.testing.assert_array_equal(routed, saved_logits)
            np.testing.assert_allclose(depth.mean(), reference_calls, atol=1e-6)
            np.testing.assert_allclose(depth.mean(), metrics["test"]["avg_optional_calls"], atol=1e-6)
            prediction = routed.argmax(1)
            accuracy = float(np.mean(prediction == labels))
            np.testing.assert_allclose(accuracy, metrics["test"]["accuracy"], atol=1e-12)
            n = len(labels)
            stop_counts = np.bincount(depth, minlength=4)
            reach_counts = reached.sum(0)
            assert int(stop_counts.sum()) == n and int(reach_counts[0]) == n
            assert np.all(np.diff(reach_counts) <= 0)
            correct_counts = [int(np.sum((depth == j) & (prediction == labels))) for j in range(4)]
            run = {
                "seed": seed, "n_test": n, "thresholds": threshold,
                "stop_counts": stop_counts.tolist(),
                "stop_pct": (100 * stop_counts / n).tolist(),
                "reach_counts": reach_counts.tolist(),
                "reach_pct": (100 * reach_counts / n).tolist(),
                "conditional_stop_pct": [
                    float(100 * stop_counts[j] / reach_counts[j]) if reach_counts[j] else None
                    for j in range(4)
                ],
                "stop_correct_counts": correct_counts,
                "stop_accuracy_pct": [
                    float(100 * correct_counts[j] / stop_counts[j]) if stop_counts[j] else None
                    for j in range(4)
                ],
                "accuracy_pct": 100 * accuracy,
                "avg_optional_calls": float(depth.mean()),
                "avg_total_agents": float(1 + depth.mean()),
                "cache_sha256": sha256(base / "stage_predictions.npz"),
                "metrics_sha256": sha256(metrics_path),
                "all_checks": "passed",
            }
            runs.append(run)
            seed_depths.append(depth)
            row = {"dataset": dataset, "seed": seed, "n_test": n,
                   "tau1": threshold[0], "tau2": threshold[1], "tau3": threshold[2]}
            for j, agent in enumerate(AGENTS):
                row[f"stop_{agent.lower()}_count"] = int(stop_counts[j])
                row[f"stop_{agent.lower()}_pct"] = float(100 * stop_counts[j] / n)
                row[f"reach_{agent.lower()}_pct"] = float(100 * reach_counts[j] / n)
            row.update(accuracy_pct=100 * accuracy, avg_optional_calls=float(depth.mean()))
            per_seed_rows.append(row)
            for i in range(n):
                j = int(depth[i])
                sample_rows.append({
                    "dataset": dataset, "seed": seed, "sample_id": str(ids[i]),
                    "true_label": int(labels[i]), "pred_label": int(prediction[i]),
                    "exit_index": j, "exit_agent": AGENTS[j],
                    "exit_confidence": float(confidence[i]),
                    "reason": "terminal_stage" if j == 3 else "confidence_ge_threshold",
                    "exit_threshold": threshold[j] if j < 3 else "",
                    "optional_calls": j,
                    **{f"reached_{a.lower()}": int(reached[i, k]) for k, a in enumerate(AGENTS)},
                })
        counts = np.array([r["stop_counts"] for r in runs])
        reach_counts = np.array([r["reach_counts"] for r in runs])
        stop_pct = np.array([r["stop_pct"] for r in runs])
        reach_pct = np.array([r["reach_pct"] for r in runs])
        total_evaluations = int(counts.sum())
        aggregate = {
            "n_test_unique": int(len(ref_ids)),
            "n_seeds": len(runs),
            "n_sample_model_evaluations": total_evaluations,
            "per_seed": runs,
            "stop_mean_pct": stop_pct.mean(0).tolist(),
            "stop_std_pct": stop_pct.std(0, ddof=1).tolist(),
            "reach_mean_pct": reach_pct.mean(0).tolist(),
            "reach_std_pct": reach_pct.std(0, ddof=1).tolist(),
            "pooled_stop_counts": counts.sum(0).tolist(),
            "pooled_reach_counts": reach_counts.sum(0).tolist(),
            "accuracy_pct": summary_stat([r["accuracy_pct"] for r in runs]),
            "avg_optional_calls": summary_stat([r["avg_optional_calls"] for r in runs]),
            "avg_total_agents": summary_stat([r["avg_total_agents"] for r in runs]),
        }
        # Supplementary: a full 10-model committee has no single shared STOP.
        # Its logical completion depth is the deepest of its member paths.
        deepest = np.stack(seed_depths).max(0)
        aggregate["committee_member_max_depth_stop_counts"] = np.bincount(deepest, minlength=4).tolist()
        for j, agent in enumerate(AGENTS):
            pooled_correct = sum(r["stop_correct_counts"][j] for r in runs)
            total_stop = int(counts[:, j].sum())
            stage_rows.append({
                "dataset": dataset, "agent": agent, "n_unique_test": len(ref_ids),
                "n_seeds": len(runs), "stop_mean_pct": float(stop_pct[:, j].mean()),
                "stop_std_pct": float(stop_pct[:, j].std(ddof=1)),
                "reach_mean_pct": float(reach_pct[:, j].mean()),
                "reach_std_pct": float(reach_pct[:, j].std(ddof=1)),
                "pooled_stop_count": total_stop,
                "pooled_reach_count": int(reach_counts[:, j].sum()),
                "pooled_conditional_stop_pct": float(100 * total_stop / reach_counts[:, j].sum())
                    if reach_counts[:, j].sum() else None,
                "pooled_exit_accuracy_pct": 100 * pooled_correct / total_stop if total_stop else None,
            })
        np.testing.assert_allclose(stop_pct.sum(1), 100)
        np.testing.assert_allclose(np.mean([r["avg_optional_calls"] for r in runs]),
                                   reach_pct[:, 1:].mean(0).sum() / 100)
        audit["datasets"][dataset] = aggregate
        print(json.dumps({"dataset": dataset, **{k: v for k, v in aggregate.items() if k != "per_seed"}}, indent=2), flush=True)
    (output / "stopping_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "stopping_by_seed.csv", per_seed_rows)
    write_csv(output / "stopping_by_agent.csv", stage_rows)
    write_csv(output / "stopping_sample_traces.csv", sample_rows)
    print(f"All 20 cached predictions reproduced exactly. Results saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/strong_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stopping_depth"))
    parser.add_argument("--datasets", nargs="+", default=["weibo", "pheme"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    analyze(parser.parse_args())
