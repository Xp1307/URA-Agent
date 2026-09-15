"""Validation-selected accuracy/computation curves for fixed URA-Agent runs.

All operating policies are selected on validation labels.  Test labels are
used only once to report accuracy/F1 and never to select thresholds.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


SEEDS = (13, 21, 42, 55, 67, 73, 87, 101, 123, 2026)
POLICY_LABELS = {
    "random": "Random routing",
    "uncertainty": "Uncertainty-only",
    "utility": "Utility-only",
    "hybrid": "Uncertainty + Utility",
}


def metrics(labels, logits):
    prediction = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def entropy(logits):
    shifted = logits - logits.max(1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(1, keepdims=True)
    return -(probs * np.log(np.maximum(probs, 1e-12))).sum(1) / math.log(logits.shape[1])


def confidence(logits):
    shifted = logits - logits.max(1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(1, keepdims=True)
    return probs.max(1)


def route(stages, route_probs, policy, params, random_uniform=None):
    n = len(stages)
    current = stages[:, 0].copy()
    alive = np.ones(n, dtype=bool)
    calls = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for j in range(3):
        if policy == "uncertainty":
            call = alive & (confidence(current) < params["tau"][j])
        elif policy == "utility":
            call = alive & (route_probs[:, j] >= params["eta"][j])
        elif policy == "hybrid":
            call = (
                alive
                & (confidence(current) < params["tau"][j])
                & (route_probs[:, j] >= params["eta"])
            )
        elif policy == "random":
            if random_uniform is None:
                raise ValueError("random_uniform is required for random routing")
            call = alive & (random_uniform[:, j] < params["q"])
        else:
            raise ValueError(policy)
        current[call] = stages[call, j + 1]
        calls[call] += 1
        depth[call] = j + 1
        alive = call
    return current, float(1 + calls.mean()), depth


def candidate_parameters(policy):
    if policy == "uncertainty":
        grid = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01)
        for values in itertools.product(grid, repeat=3):
            yield {"tau": list(values)}
    elif policy == "utility":
        grid = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01)
        for values in itertools.product(grid, repeat=3):
            yield {"eta": list(values)}
    elif policy == "hybrid":
        tau_grid = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01)
        eta_grid = (0.00, 0.20, 0.40, 0.50, 0.60, 0.80, 1.01)
        for values in itertools.product(tau_grid, repeat=3):
            for eta in eta_grid:
                yield {"tau": list(values), "eta": eta}
    else:
        raise ValueError(policy)


def enumerate_validation(stages, probs, labels, policy):
    candidates = []
    for params in candidate_parameters(policy):
        logits, agents, _ = route(stages, probs, policy, params)
        score = metrics(labels, logits)
        candidates.append({"params": params, "agents": agents, **score})
    return candidates


def select_candidate(candidates, budget):
    feasible = [c for c in candidates if c["agents"] <= budget + 1e-12]
    if not feasible:
        feasible = [min(candidates, key=lambda c: c["agents"])]
    return max(
        feasible,
        key=lambda c: (c["accuracy"], c["macro_f1"], -c["agents"]),
    )


def solve_random_q(budget):
    if budget <= 1:
        return 0.0
    if budget >= 4:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2
        value = 1 + mid + mid**2 + mid**3
        if value < budget:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cross_entropy_per_sample(logits, labels):
    maximum = logits.max(1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(1))
    return logsumexp - logits[np.arange(len(labels)), labels]


def utility_targets(stages, route_probs, labels):
    stage0, image, grounding, consistency = [stages[:, j] for j in range(4)]
    gate1 = route_probs[:, 0] * entropy(stage0)
    stage1 = (1 - gate1[:, None]) * stage0 + gate1[:, None] * image
    gate2 = route_probs[:, 1] * entropy(stage1)
    stage2 = (1 - gate2[:, None]) * stage1 + gate2[:, None] * grounding
    previous = (stage0, stage1, stage2)
    experts = (image, grounding, consistency)
    return np.stack(
        [
            cross_entropy_per_sample(expert, labels)
            < cross_entropy_per_sample(base, labels)
            for base, expert in zip(previous, experts)
        ],
        axis=1,
    ).astype(np.int64)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utility_metrics(dataset, seed, split, stages, probs, labels):
    targets = utility_targets(stages, probs, labels)
    rows = []
    for j, transition in enumerate(("Text→Vision", "Vision→Grounding", "Grounding→Consistency")):
        target = targets[:, j]
        score = probs[:, j]
        row = {
            "dataset": dataset,
            "seed": seed,
            "split": split,
            "transition": transition,
            "positive_rate": float(target.mean()),
            "auroc": "",
            "auprc": "",
        }
        if np.unique(target).size == 2:
            row["auroc"] = float(roc_auc_score(target, score))
            row["auprc"] = float(average_precision_score(target, score))
        rows.append(row)
    return rows


def analyze(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    budgets = np.asarray(args.budgets, dtype=float)
    by_seed, utility_rows = [], []
    cached = {}
    for dataset_index, dataset in enumerate(args.datasets):
        for seed in args.seeds:
            run = args.score_dir / "scores" / dataset / f"seed_{seed}"
            values = {}
            for split in ("val", "test"):
                with np.load(run / f"{split}_router_scores.npz", allow_pickle=False) as data:
                    values[split] = {key: data[key] for key in data.files}
                utility_rows.extend(
                    utility_metrics(
                        dataset,
                        seed,
                        split,
                        values[split]["stage_logits"],
                        values[split]["route_probs"],
                        values[split]["labels"],
                    )
                )
            cached[(dataset, seed)] = values
            for policy in ("uncertainty", "utility", "hybrid"):
                val = values["val"]
                candidates = enumerate_validation(
                    val["stage_logits"], val["route_probs"], val["labels"], policy
                )
                for budget in budgets:
                    chosen = select_candidate(candidates, float(budget))
                    test = values["test"]
                    logits, agents, depth = route(
                        test["stage_logits"], test["route_probs"], policy, chosen["params"]
                    )
                    score = metrics(test["labels"], logits)
                    by_seed.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "policy_label": POLICY_LABELS[policy],
                            "budget": float(budget),
                            "seed": seed,
                            "validation_accuracy": chosen["accuracy"],
                            "validation_macro_f1": chosen["macro_f1"],
                            "validation_avg_agents": chosen["agents"],
                            "test_accuracy": score["accuracy"],
                            "test_macro_f1": score["macro_f1"],
                            "test_avg_agents": agents,
                            "parameters": json.dumps(chosen["params"], separators=(",", ":")),
                            "test_text_exit_pct": float(100 * np.mean(depth == 0)),
                            "test_vision_exit_pct": float(100 * np.mean(depth == 1)),
                            "test_grounding_exit_pct": float(100 * np.mean(depth == 2)),
                            "test_consistency_exit_pct": float(100 * np.mean(depth == 3)),
                        }
                    )

            test = values["test"]
            for budget_index, budget in enumerate(budgets):
                q = solve_random_q(float(budget))
                rep_scores, rep_agents = [], []
                for repetition in range(args.random_repetitions):
                    rng_seed = (
                        seed * 1_000_003
                        + dataset_index * 100_003
                        + budget_index * 1_009
                        + repetition
                    )
                    rng = np.random.default_rng(rng_seed)
                    uniforms = rng.random((len(test["labels"]), 3))
                    logits, agents, _ = route(
                        test["stage_logits"],
                        test["route_probs"],
                        "random",
                        {"q": q},
                        random_uniform=uniforms,
                    )
                    rep_scores.append(metrics(test["labels"], logits))
                    rep_agents.append(agents)
                by_seed.append(
                    {
                        "dataset": dataset,
                        "policy": "random",
                        "policy_label": POLICY_LABELS["random"],
                        "budget": float(budget),
                        "seed": seed,
                        "validation_accuracy": "",
                        "validation_macro_f1": "",
                        "validation_avg_agents": "",
                        "test_accuracy": float(np.mean([x["accuracy"] for x in rep_scores])),
                        "test_macro_f1": float(np.mean([x["macro_f1"] for x in rep_scores])),
                        "test_avg_agents": float(np.mean(rep_agents)),
                        "parameters": json.dumps(
                            {"q": q, "repetitions": args.random_repetitions},
                            separators=(",", ":"),
                        ),
                        "test_text_exit_pct": "",
                        "test_vision_exit_pct": "",
                        "test_grounding_exit_pct": "",
                        "test_consistency_exit_pct": "",
                    }
                )

    summary = []
    for dataset in args.datasets:
        for policy in POLICY_LABELS:
            for budget in budgets:
                selected = [
                    r
                    for r in by_seed
                    if r["dataset"] == dataset
                    and r["policy"] == policy
                    and r["budget"] == float(budget)
                ]
                summary.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "budget": float(budget),
                        "n_seeds": len(selected),
                        "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in selected])),
                        "test_accuracy_std": float(np.std([r["test_accuracy"] for r in selected], ddof=1)),
                        "test_macro_f1_mean": float(np.mean([r["test_macro_f1"] for r in selected])),
                        "test_macro_f1_std": float(np.std([r["test_macro_f1"] for r in selected], ddof=1)),
                        "test_avg_agents_mean": float(np.mean([r["test_avg_agents"] for r in selected])),
                        "test_avg_agents_std": float(np.std([r["test_avg_agents"] for r in selected], ddof=1)),
                    }
                )

    write_csv(output / "pareto_by_seed.csv", by_seed)
    write_csv(output / "pareto_summary.csv", summary)
    write_csv(output / "routing_utility_metrics.csv", utility_rows)
    metadata = {
        "selection": "For every seed and compute budget, choose thresholds by validation accuracy; tie-break with validation Macro-F1 and lower validation cost.",
        "test_label_use": "Metric evaluation only; never policy selection.",
        "computation_axis": "Mean number of logically active agents (Text is mandatory; Vision/Grounding/Consistency are optional).",
        "compute_caveat": "Logical conditional computation. The current forward pass is eager and this is not a measured latency/FLOP claim.",
        "random_baseline": f"Sequential equal-probability random calls; {args.random_repetitions} Monte Carlo repetitions per seed/budget.",
        "budgets": budgets.tolist(),
        "seeds": args.seeds,
        "datasets": args.datasets,
    }
    (output / "pareto_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot(summary, args.datasets, output)
    report(summary, utility_rows, args.datasets, output)


def plot(summary, datasets, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "random": "#9CA3AF",
        "uncertainty": "#2563EB",
        "utility": "#F59E0B",
        "hybrid": "#DC2626",
    }
    markers = {"random": "o", "uncertainty": "s", "utility": "^", "hybrid": "D"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.6, 3.65), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        values_all = []
        for policy in ("random", "uncertainty", "utility", "hybrid"):
            values = [r for r in summary if r["dataset"] == dataset and r["policy"] == policy]
            values.sort(key=lambda r: r["budget"])
            x = np.asarray([r["test_avg_agents_mean"] for r in values])
            y = 100 * np.asarray([r["test_accuracy_mean"] for r in values])
            e = 100 * np.asarray([r["test_accuracy_std"] for r in values])
            values_all.extend((y - e).tolist())
            values_all.extend((y + e).tolist())
            axis.plot(
                x,
                y,
                color=colors[policy],
                marker=markers[policy],
                markersize=4.3,
                linewidth=1.8,
                label=POLICY_LABELS[policy],
                zorder=3,
            )
            axis.fill_between(x, y - e, y + e, color=colors[policy], alpha=0.10, linewidth=0)
        axis.set_title("Weibo" if dataset == "weibo" else "PHEME", fontweight="bold")
        axis.set_xlabel("Average Number of Active Agents")
        axis.set_xlim(0.94, 4.06)
        axis.set_xticks((1, 1.5, 2, 2.5, 3, 3.5, 4))
        low, high = min(values_all), max(values_all)
        padding = max(0.8, 0.08 * (high - low))
        axis.set_ylim(math.floor(low - padding), math.ceil(high + padding))
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy–Computation Pareto Curve", y=1.11, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output / f"accuracy_computation_pareto.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def report(summary, utility_rows, datasets, output):
    lines = [
        "# URA-Agent Accuracy–Computation Pareto Experiment",
        "",
        "Operating points are selected on the validation split independently for each seed and budget; test labels are used only for final evaluation.",
        "The x-axis reports logical active agents rather than measured latency because the present forward implementation eagerly computes all experts.",
        "",
    ]
    for dataset in datasets:
        lines.extend([f"## {dataset.upper()}", "", "| Policy | Best test Acc. | Active agents at that point |", "|---|---:|---:|"])
        for policy in ("random", "uncertainty", "utility", "hybrid"):
            values = [r for r in summary if r["dataset"] == dataset and r["policy"] == policy]
            best = max(values, key=lambda r: r["test_accuracy_mean"])
            lines.append(
                f"| {POLICY_LABELS[policy]} | {100*best['test_accuracy_mean']:.2f}±{100*best['test_accuracy_std']:.2f} | "
                f"{best['test_avg_agents_mean']:.3f}±{best['test_avg_agents_std']:.3f} |"
            )
        test_utility = [r for r in utility_rows if r["dataset"] == dataset and r["split"] == "test"]
        auprc_values = [float(r["auprc"]) for r in test_utility if r["auprc"] != ""]
        lines.extend(
            [
                "",
                f"Mean test utility AUPRC across valid seed/transition pairs: {np.mean(auprc_values):.4f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `accuracy_computation_pareto.png/.svg/.pdf`: publication figure.",
            "- `pareto_by_seed.csv`: validation-selected thresholds and test results for every seed/budget.",
            "- `pareto_summary.csv`: mean±sample-standard-deviation source data for the figure.",
            "- `routing_utility_metrics.csv`: router AUROC/AUPRC against the training-time utility definition.",
            "- `pareto_metadata.json`: evaluation protocol and caveats.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["weibo", "pheme"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0],
    )
    parser.add_argument("--random-repetitions", type=int, default=200)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
