"""Run a reproducible sequence of strong baselines for one dataset/GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=("weibo", "pheme"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "text_ft",
            "text_refine",
            "joint_ft",
            "joint_refine",
            "concat_ft",
            "cafe_adapted",
            "mcot",
            "ura_agent",
        ],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--vision-layers", type=int, default=2)
    parser.add_argument("--pooling", choices=("cls", "cls_mean"), default="cls")
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--merge-val", action="store_true")
    parser.add_argument(
        "--agent-text-source",
        choices=("auto", "text_ft", "text_refine"),
        default="auto",
    )
    parser.add_argument("--output-dir", default="artifacts/strong_results")
    parser.add_argument("--init-results-dir", default="artifacts/strong_results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    log_dir = args.project_root / "artifacts" / "strong_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for method in args.methods:
            result = (
                args.project_root
                / args.output_dir
                / args.dataset
                / method
                / f"seed_{seed}"
                / "metrics.json"
            )
            if result.exists() and not args.force:
                print(f"SKIP existing {args.dataset}/{method}/seed_{seed}", flush=True)
                continue
            log_path = log_dir / f"{args.dataset}_{method}_seed{seed}.log"
            command = [
                sys.executable,
                str(args.project_root / "src" / "train_strong.py"),
                "--project-root",
                str(args.project_root),
                "--dataset",
                args.dataset,
                "--method",
                method,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--num-workers",
                str(args.num_workers),
                "--encoder-lr",
                str(args.encoder_lr),
                "--head-lr",
                str(args.head_lr),
                "--text-layers",
                str(args.text_layers),
                "--vision-layers",
                str(args.vision_layers),
                "--pooling",
                args.pooling,
                "--class-weighting",
                args.class_weighting,
                "--selection-metric",
                args.selection_metric,
                "--agent-text-source",
                args.agent_text_source,
                "--output-dir",
                args.output_dir,
                "--init-results-dir",
                args.init_results_dir,
            ]
            if args.merge_val:
                command.append("--merge-val")
            print(f"RUN {' '.join(command)}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=args.project_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise SystemExit(f"FAILED {args.dataset}/{method}/seed_{seed}; see {log_path}")
            print(f"DONE {args.dataset}/{method}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
