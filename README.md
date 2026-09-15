# URA-Agent

Official research implementation of **URA-Agent: Uncertainty-Guided Agent Routing for Multimodal Fake News Detection**.

URA-Agent treats multimodal fake-news detection as conditional evidence acquisition. It begins with textual evidence and invokes the Vision, Grounding, and Consistency agents only when the current prediction remains uncertain. During training, utility targets supervise the routing modules to estimate whether the next specialist can improve the current routed prediction.

## Method overview

The agents are evaluated in the following order:

1. **Text Agent** encodes the post text with BERT (Weibo) or RoBERTa (PHEME).
2. **Vision Agent** extracts visual evidence with ViT.
3. **Grounding Agent** jointly encodes the post text and an offline MLLM-generated image description.
4. **Consistency Agent** models support/conflict relations among text, image, and grounding evidence.

The main hard-inference policy calls the next agent when the current prediction confidence is below a validation-selected stage threshold:

```text
call A_(j+1) iff max_k p_k^(j) < tau_j
```

The learned utility scores supervise differentiable routing during training. The optional Pareto analysis additionally evaluates a hybrid policy:

```text
call A_(j+1) iff max_k p_k^(j) < tau_j and r_j >= eta
```

This distinction is intentional: the first rule reproduces the core hard-routing pipeline, whereas the second is an additional uncertainty--utility trade-off analysis.

## Repository structure

```text
URA-Agent/
├── configs/
│   └── paper_setup.json
├── scripts/
│   ├── run_pheme_10seeds.sh
│   └── run_weibo_10seeds.sh
├── src/
│   ├── train_strong.py
│   ├── run_suite.py
│   ├── summarize_results.py
│   ├── summarize_ten_runs.py
│   ├── evaluate_ablation.py
│   ├── evaluate_consensus.py
│   ├── evaluate_stage_policy.py
│   ├── analyze_stopping_depth.py
│   ├── export_router_scores.py
│   ├── analyze_accuracy_computation_pareto.py
│   └── plot_pheme_committee_pareto.py
├── .gitignore
├── README.md
└── requirements.txt
```

Datasets, pretrained models, checkpoints, predictions, logs, and generated result files are deliberately excluded.

## Environment

The released experiments use Python 3.10 and PyTorch with CUDA. Create an isolated environment and install the dependencies:

```bash
conda create -n ura-agent python=3.10 -y
conda activate ura-agent
pip install -r requirements.txt
```

The code loads all Hugging Face backbones with `local_files_only=True`. Download or cache these models before training:

- `bert-base-chinese` for Weibo text;
- `roberta-base` for PHEME text;
- `google/vit-base-patch16-224-in21k` for images.

## Data preparation

Place the data under the repository root as follows:

```text
datasets/
├── weibo/
│   ├── data_json/{train,val,test}_imdes.json
│   └── weibo_images/images_all/
└── pheme/
    ├── data_json/{train,val,test}_imdes.json
    └── pheme_images/images_all/
```

Each split may be a JSON list or a dictionary of records. A record must contain:

```json
{
  "tweet_id": "sample_id",
  "tweet_content": "post text",
  "img_id": "image_name.jpg",
  "tweet_label": 1,
  "Qwen3-VL-8B-Instruct_generation": "offline MLLM image description"
}
```

`tweet_label=0` denotes real news and `tweet_label=1` denotes fake news. The MLLM description is generated offline and is treated as an input field; no MLLM is called by the training pipeline.

## Training

URA-Agent is initialized from trained Text and Grounding specialists. Train the prerequisite models before the final agent:

```bash
python src/run_suite.py \
  --project-root . \
  --dataset weibo \
  --device cuda:0 \
  --seeds 13 \
  --methods text_ft joint_ft joint_refine ura_agent \
  --text-layers 4 \
  --vision-layers 2 \
  --pooling cls_mean \
  --class-weighting balanced \
  --selection-metric accuracy \
  --agent-text-source text_ft
```

For PHEME, use `--methods text_ft text_refine joint_ft joint_refine ura_agent` and `--agent-text-source text_refine`.

The ten-seed launchers reproduce the documented training setup:

```bash
bash scripts/run_weibo_10seeds.sh cuda:0
bash scripts/run_pheme_10seeds.sh cuda:0
```

Outputs are written to `artifacts/strong_results/<dataset>/<method>/seed_<seed>/`. Each run stores its configuration, validation history, metrics, checkpoint, routed predictions, and per-stage logits.

## Evaluation and analysis

Summarize ten independent runs:

```bash
python src/summarize_ten_runs.py \
  --results-root artifacts/strong_results \
  --seeds 13 21 42 55 67 73 87 101 123 2026 \
  --output-json outputs/ten_seed_summary.json \
  --output-md outputs/ten_seed_summary.md
```

Run the component ablations (`Text-only`, `w/o Vision`, `w/o Grounding`, `w/o Consistency`, `w/o Adaptive Router`, and the full model):

```bash
python src/evaluate_ablation.py \
  --project-root . \
  --results-root artifacts/strong_results \
  --output-dir outputs/ablation
```

Audit the fixed confidence-routing paths and export stage-reach statistics:

```bash
python src/analyze_stopping_depth.py \
  --project-root . \
  --results-root artifacts/strong_results \
  --output-dir outputs/stopping_depth
```

Export learned router utilities and generate the Accuracy--Activation Pareto analysis:

```bash
python src/export_router_scores.py \
  --project-root . \
  --results-root artifacts/strong_results \
  --output-dir outputs/router_scores \
  --device cuda:0

python src/analyze_accuracy_computation_pareto.py \
  --score-dir outputs/router_scores/scores \
  --output-dir outputs/pareto
```

The stage-reach analysis counts logical agent activations.

## Reproducibility notes

- All routing thresholds and model-selection decisions are selected on the validation split; test labels are never used for routing.
- Mean and standard deviation use the ten seeds listed in `configs/paper_setup.json`.
- `evaluate_ablation.py` emits both independent per-seed statistics and a separately labeled fixed-policy leave-one-member-out committee stability analysis. These protocols should not be reported interchangeably.
- Random seeds and deterministic CuDNN settings are fixed in `src/train_strong.py`; exact values may still vary across CUDA, PyTorch, and hardware versions.

## Citation

Citation metadata will be added after publication.
