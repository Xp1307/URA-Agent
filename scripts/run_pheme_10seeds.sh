#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-cuda:0}"

cd "$ROOT"
"$PYTHON_BIN" -u src/run_suite.py \
  --project-root "$ROOT" \
  --dataset pheme \
  --device "$DEVICE" \
  --seeds 13 21 42 55 67 73 87 101 123 2026 \
  --methods text_ft text_refine joint_ft joint_refine ura_agent \
  --epochs 15 \
  --batch-size 16 \
  --patience 4 \
  --num-workers 4 \
  --encoder-lr 1e-5 \
  --head-lr 2e-4 \
  --text-layers 4 \
  --vision-layers 2 \
  --pooling cls_mean \
  --class-weighting balanced \
  --selection-metric accuracy \
  --agent-text-source text_refine \
  --output-dir artifacts/strong_results \
  --init-results-dir artifacts/strong_results

