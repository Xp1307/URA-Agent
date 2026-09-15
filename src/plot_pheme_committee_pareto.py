"""Plot a PHEME-only committee accuracy/computation curve.

This script keeps the evaluation unit consistent with the reported
90.29+/-0.18 result: validation-selected committee aggregation followed by
fixed-policy leave-one-member-out perturbation.  It never edits measured
accuracies or shifts a curve to a requested target.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


SEEDS = (13, 21, 42, 55, 67, 73, 87, 101, 123, 2026)
POLICIES = ("random", "uncertainty", "utility", "hybrid")
LABELS = {
    "random": "Random routing",
    "uncertainty": "Uncertainty-only",
    "utility": "Utility-only",
    "hybrid": "Uncertainty + Utility",
    "full_ura": "Uncertainty + Utility (URA-Agent)",
}


def load_analysis_module(path: Path):
    spec = importlib.util.spec_from_file_location("pareto_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def softmax(logits):
    shifted = logits - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(1, keepdims=True)


def select_bias(labels, score, train_positive_prior):
    best = None
    for bias in np.linspace(-1.0, 1.0, 401):
        accuracy = float(accuracy_score(labels, score + bias > 0))
        if train_positive_prior < labels.mean():
            prior_tiebreak = -float(bias)
        elif train_positive_prior > labels.mean():
            prior_tiebreak = float(bias)
        else:
            prior_tiebreak = -abs(float(bias))
        candidate = (accuracy, prior_tiebreak, -abs(float(bias)), float(bias))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], best[0]


def committee_fixed_policy(labels_val, labels_test, val_logits, test_logits, prior):
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
            [(x - v.mean()) / (v.std() + 1e-8) for x, v in zip(raw_test, raw_val)],
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
        bias, accuracy = select_bias(labels_val, np.mean(val_scores, axis=0), prior)
        candidates.append((accuracy, priority[name], name, bias))
    candidates.sort(reverse=True)
    _, _, family, fixed_bias = candidates[0]
    _, test_scores = families[family]
    accuracies = []
    for held_out in range(len(test_scores)):
        keep = [i for i in range(len(test_scores)) if i != held_out]
        score = np.mean([test_scores[i] for i in keep], axis=0) + fixed_bias
        accuracies.append(float(accuracy_score(labels_test, score > 0)))
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "selected_family": family,
        "selected_bias": float(fixed_bias),
        "member_perturbation_accuracies": accuracies,
    }


def load_scores(score_root, seed):
    result = {}
    run = score_root / "scores" / "pheme" / f"seed_{seed}"
    for split in ("val", "test"):
        with np.load(run / f"{split}_router_scores.npz", allow_pickle=False) as data:
            result[split] = {key: data[key] for key in data.files}
    return result


def select_budgeted_full_policy(analysis, stages, route_probs, labels, budget):
    """Match the original Full URA validation tuning, adding a cost ceiling."""
    grid = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01)
    best = None
    for thresholds in itertools.product(grid, repeat=3):
        params = {"tau": list(thresholds)}
        logits, active_agents, _ = analysis.route(
            stages, route_probs, "uncertainty", params
        )
        if active_agents > budget + 1e-12:
            continue
        accuracy = float(accuracy_score(labels, logits.argmax(1)))
        candidate = (accuracy, -active_agents, params)
        # The original implementation breaks ties only with lower calls and
        # keeps the first threshold tuple encountered thereafter.
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No feasible Full URA policy for budget={budget}")
    return best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reported-full-json", type=Path)
    parser.add_argument("--train-positive-prior", type=float, default=0.29249291784702547)
    parser.add_argument("--random-repetitions", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_module(args.analysis_script)

    with args.source_csv.open(encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    source = {
        (row["policy"], float(row["budget"]), int(row["seed"])): row
        for row in source_rows
        if row["dataset"] == "pheme"
    }
    budgets = sorted({float(row["budget"]) for row in source_rows if row["dataset"] == "pheme"})
    cache = {seed: load_scores(args.score_dir, seed) for seed in SEEDS}
    labels_val = cache[SEEDS[0]]["val"]["labels"]
    labels_test = cache[SEEDS[0]]["test"]["labels"]
    rows = []

    for policy in POLICIES:
        for budget_index, budget in enumerate(budgets):
            if policy != "random":
                val_logits, test_logits, costs = [], [], []
                for seed in SEEDS:
                    params = json.loads(source[(policy, budget, seed)]["parameters"])
                    val = cache[seed]["val"]
                    test = cache[seed]["test"]
                    routed_val, _, _ = analysis.route(
                        val["stage_logits"], val["route_probs"], policy, params
                    )
                    routed_test, test_agents, _ = analysis.route(
                        test["stage_logits"], test["route_probs"], policy, params
                    )
                    val_logits.append(routed_val)
                    test_logits.append(routed_test)
                    costs.append(test_agents)
                result = committee_fixed_policy(
                    labels_val,
                    labels_test,
                    val_logits,
                    test_logits,
                    args.train_positive_prior,
                )
                rows.append(
                    {
                        "dataset": "pheme",
                        "policy": policy,
                        "policy_label": LABELS[policy],
                        "budget": budget,
                        "active_agents_mean": float(np.mean(costs)),
                        "active_agents_std": float(np.std(costs, ddof=1)),
                        **result,
                    }
                )
            else:
                repetition_results, repetition_costs = [], []
                for repetition in range(args.random_repetitions):
                    val_logits, test_logits, costs = [], [], []
                    for seed in SEEDS:
                        q = json.loads(source[(policy, budget, seed)]["parameters"])["q"]
                        val = cache[seed]["val"]
                        test = cache[seed]["test"]
                        val_rng = np.random.default_rng(
                            seed * 1_000_003 + budget_index * 1_009 + repetition
                        )
                        test_rng = np.random.default_rng(
                            seed * 1_000_003 + budget_index * 1_009 + repetition + 50_000_003
                        )
                        routed_val, _, _ = analysis.route(
                            val["stage_logits"],
                            val["route_probs"],
                            "random",
                            {"q": q},
                            random_uniform=val_rng.random((len(labels_val), 3)),
                        )
                        routed_test, test_agents, _ = analysis.route(
                            test["stage_logits"],
                            test["route_probs"],
                            "random",
                            {"q": q},
                            random_uniform=test_rng.random((len(labels_test), 3)),
                        )
                        val_logits.append(routed_val)
                        test_logits.append(routed_test)
                        costs.append(test_agents)
                    repetition_results.append(
                        committee_fixed_policy(
                            labels_val,
                            labels_test,
                            val_logits,
                            test_logits,
                            args.train_positive_prior,
                        )
                    )
                    repetition_costs.append(float(np.mean(costs)))
                # Random routing variation and leave-one-member-out variation are
                # both reflected in the pooled standard deviation.
                pooled = np.asarray(
                    [
                        value
                        for result in repetition_results
                        for value in result["member_perturbation_accuracies"]
                    ]
                )
                rows.append(
                    {
                        "dataset": "pheme",
                        "policy": policy,
                        "policy_label": LABELS[policy],
                        "budget": budget,
                        "active_agents_mean": float(np.mean(repetition_costs)),
                        "active_agents_std": float(np.std(repetition_costs, ddof=1)),
                        "accuracy_mean": float(pooled.mean()),
                        "accuracy_std": float(pooled.std(ddof=1)),
                        "selected_family": "varies_by_repetition",
                        "selected_bias": "",
                        "member_perturbation_accuracies": "",
                    }
                )

    # Full URA-Agent curve under the exact original confidence-routing family.
    # The model itself retains utility-supervised training; each budget adds a
    # validation-only ceiling on the mean number of active agents.  At budget
    # 4.0 this reduces exactly to the reported Full URA operating policy.
    for budget in budgets:
        val_logits, test_logits, costs = [], [], []
        for seed in SEEDS:
            val = cache[seed]["val"]
            test = cache[seed]["test"]
            params = select_budgeted_full_policy(
                analysis,
                val["stage_logits"],
                val["route_probs"],
                val["labels"],
                budget,
            )
            routed_val, _, _ = analysis.route(
                val["stage_logits"], val["route_probs"], "uncertainty", params
            )
            routed_test, test_agents, _ = analysis.route(
                test["stage_logits"], test["route_probs"], "uncertainty", params
            )
            val_logits.append(routed_val)
            test_logits.append(routed_test)
            costs.append(test_agents)
        result = committee_fixed_policy(
            labels_val,
            labels_test,
            val_logits,
            test_logits,
            args.train_positive_prior,
        )
        rows.append(
            {
                "dataset": "pheme",
                "policy": "full_ura",
                "policy_label": LABELS["full_ura"],
                "budget": budget,
                "active_agents_mean": float(np.mean(costs)),
                "active_agents_std": float(np.std(costs, ddof=1)),
                **result,
            }
        )

    output_csv = args.output_dir / "pheme_committee_pareto.csv"
    serializable = []
    for row in rows:
        copy = dict(row)
        if isinstance(copy["member_perturbation_accuracies"], list):
            copy["member_perturbation_accuracies"] = json.dumps(
                copy["member_perturbation_accuracies"], separators=(",", ":")
            )
        serializable.append(copy)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)

    reported_full = None
    if args.reported_full_json is not None:
        reported_payload = json.loads(args.reported_full_json.read_text(encoding="utf-8"))
        full = reported_payload["Full URA-Agent"]
        reported_full = {
            "active_agents_mean": 1.0 + float(
                np.mean([run["calls"] for run in full["per_seed"]])
            ),
            "active_agents_std": float(
                np.std([run["calls"] for run in full["per_seed"]], ddof=1)
            ),
            "accuracy_mean": float(full["committee"]["fixed_policy_mean"]["accuracy"]),
            "accuracy_std": float(full["committee"]["fixed_policy_std"]["accuracy"]),
            "source": str(args.reported_full_json.resolve()),
            "protocol": "reported fixed-policy leave-one-member-out committee stability",
        }
        (args.output_dir / "pheme_reported_full_ura_point.json").write_text(
            json.dumps(reported_full, indent=2), encoding="utf-8"
        )
        final_full = max(
            (row for row in rows if row["policy"] == "full_ura"),
            key=lambda row: row["budget"],
        )
        if not np.isclose(
            final_full["accuracy_mean"], reported_full["accuracy_mean"], atol=1e-12
        ) or not np.isclose(
            final_full["accuracy_std"], reported_full["accuracy_std"], atol=1e-12
        ):
            raise AssertionError(
                "Budget-swept Full URA endpoint did not reproduce the reported result: "
                f"curve={final_full['accuracy_mean']}±{final_full['accuracy_std']}, "
                f"reported={reported_full['accuracy_mean']}±{reported_full['accuracy_std']}"
            )
    plot(rows, args.output_dir, reported_full)
    metadata = {
        "dataset": "PHEME",
        "evaluation_unit": "fixed-policy leave-one-member-out committee stability",
        "threshold_source": "per-seed validation-selected operating policy",
        "aggregation_selection": "validation only",
        "test_label_use": "metric evaluation only",
        "train_positive_prior": args.train_positive_prior,
        "random_repetitions": args.random_repetitions,
        "accuracy_target_was_not_imposed": True,
        "reported_full_ura_point": reported_full,
        "display_note": (
            "The red display is the validation-budgeted Full URA-Agent curve. Its unconstrained endpoint "
            "reproduces the reported result exactly. The alternative H>tau AND q>eta hybrid curve remains "
            "in pheme_committee_pareto.csv for audit."
            if reported_full is not None
            else "All four recomputed policy curves are displayed."
        ),
    }
    (args.output_dir / "pheme_committee_pareto_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def plot(rows, output, reported_full=None):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "random": "#9CA3AF",
        "uncertainty": "#2563EB",
        "utility": "#F59E0B",
        "hybrid": "#DC2626",
        "full_ura": "#DC2626",
    }
    markers = {
        "random": "o",
        "uncertainty": "s",
        "utility": "^",
        "hybrid": "D",
        "full_ura": "D",
    }
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    all_low, all_high = [], []
    displayed_policies = (
        ("random", "uncertainty", "utility", "full_ura")
        if reported_full is not None
        else POLICIES
    )
    for policy in displayed_policies:
        values = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["budget"])
        x = np.asarray([r["active_agents_mean"] for r in values])
        y = 100 * np.asarray([r["accuracy_mean"] for r in values])
        error = 100 * np.asarray([r["accuracy_std"] for r in values])
        all_low.extend((y - error).tolist())
        all_high.extend((y + error).tolist())
        axis.plot(
            x,
            y,
            color=colors[policy],
            marker=markers[policy],
            markersize=5,
            linewidth=2.0,
            label=LABELS[policy],
            zorder=3,
        )
        axis.fill_between(x, y - error, y + error, color=colors[policy], alpha=0.11, linewidth=0)
    if reported_full is not None:
        full_values = sorted(
            (row for row in rows if row["policy"] == "full_ura"),
            key=lambda row: row["budget"],
        )
        endpoint = full_values[-1]
        x = endpoint["active_agents_mean"]
        y = 100 * endpoint["accuracy_mean"]
        error = 100 * endpoint["accuracy_std"]
        axis.annotate(
            f"{y:.2f}±{error:.2f}",
            xy=(x, y),
            xytext=(12, -21),
            textcoords="offset points",
            color=colors["hybrid"],
            fontsize=9.5,
            fontweight="bold",
        )
    axis.set_xlabel("Average Number of Active Agents")
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_xlim(0.94, 4.06)
    axis.set_xticks((1, 1.5, 2, 2.5, 3, 3.5, 4))
    low, high = min(all_low), max(all_high)
    padding = max(0.4, 0.08 * (high - low))
    axis.set_ylim(math.floor(2 * (low - padding)) / 2, math.ceil(2 * (high + padding)) / 2)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("PHEME: Accuracy–Computation Pareto Curve", y=0.985, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(output / f"pheme_accuracy_computation_pareto.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
