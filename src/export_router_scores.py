"""Export learned URA-Agent router scores from fixed checkpoints.

This is an inference-only audit.  It reconstructs each trained model from the
same pretrained/source checkpoints used by ``train_strong.py``, then saves the
validation/test route probabilities without changing any model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SEEDS = (13, 21, 42, 55, 67, 73, 87, 101, 123, 2026)


def load_train_module(path: Path):
    spec = importlib.util.spec_from_file_location("ura_train_strong", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def build_model(ts, root: Path, result_root: Path, dataset: str, seed: int, device):
    run_dir = result_root / dataset / "ura_agent" / f"seed_{seed}"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = ts.URAAgent(
        dataset=dataset,
        text_layers=int(config["text_layers"]),
        vision_layers=int(config["vision_layers"]),
        pooling=config["pooling"],
    )

    text_method = config.get("agent_text_source", "text_refine")
    if text_method == "auto":
        text_method = "text_refine"
    text_path = result_root / dataset / text_method / f"seed_{seed}" / "best_trainable.pt"
    if not text_path.exists():
        text_path = result_root / dataset / "text_ft" / f"seed_{seed}" / "best_trainable.pt"
    text_state = load_checkpoint(text_path)
    mapped = {}
    for key, value in text_state.items():
        if key.startswith("backbone.text."):
            mapped[key] = value
        elif key.startswith("classifier."):
            mapped["text_head." + key.removeprefix("classifier.")] = value
    model.load_state_dict(mapped, strict=False)

    grounding_path = (
        result_root / dataset / "joint_refine" / f"seed_{seed}" / "best_trainable.pt"
    )
    if not grounding_path.exists():
        grounding_path = (
            result_root / dataset / "joint_ft" / f"seed_{seed}" / "best_trainable.pt"
        )
    grounding_state = load_checkpoint(grounding_path)
    grounding_mapped = {}
    for key, value in grounding_state.items():
        if key.startswith("backbone.text."):
            grounding_mapped[
                "grounding_text." + key.removeprefix("backbone.text.")
            ] = value
        elif key.startswith("classifier."):
            grounding_mapped[
                "grounding_head." + key.removeprefix("classifier.")
            ] = value
    model.load_state_dict(grounding_mapped, strict=False)

    ura_path = run_dir / "best_trainable.pt"
    model.load_state_dict(load_checkpoint(ura_path), strict=False)
    model.to(device).eval()
    return model, config, run_dir, text_path, grounding_path, ura_path


@torch.no_grad()
def infer_scores(ts, model, loader, device):
    ids, labels, stages, route_probs, full_logits = [], [], [], [], []
    for batch in loader:
        ids.extend(batch.pop("ids"))
        batch = ts.move_batch(batch, device)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
        labels.append(batch["labels"].cpu())
        stages.append(output["stage_logits"].float().cpu())
        route_probs.append(output["route_probs"].float().cpu())
        full_logits.append(output["logits"].float().cpu())
    return {
        "ids": np.asarray(ids),
        "labels": torch.cat(labels).numpy(),
        "stage_logits": torch.cat(stages).numpy(),
        "route_probs": torch.cat(route_probs).numpy(),
        "full_logits": torch.cat(full_logits).numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["weibo", "pheme"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.project_root.resolve()
    results = (root / args.results_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "src/train_strong.py"
    ts = load_train_module(source)
    device = torch.device(args.device)
    manifest = {
        "analysis": "Inference-only export of learned utility-router probabilities",
        "source": str(source),
        "source_sha256": sha256(source),
        "results_root": str(results),
        "device": str(device),
        "runs": [],
    }

    for dataset in args.datasets:
        collator = ts.BatchCollator(dataset, "ura_agent")
        datasets = {
            split: ts.NewsDataset(root, dataset, split) for split in ("val", "test")
        }
        loaders = {
            split: DataLoader(
                data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                collate_fn=collator,
            )
            for split, data in datasets.items()
        }
        for seed in args.seeds:
            print(f"Exporting {dataset} seed={seed}", flush=True)
            ts.set_seed(seed)
            model, config, run_dir, text_path, grounding_path, ura_path = build_model(
                ts, root, results, dataset, seed, device
            )
            run_output = output / "scores" / dataset / f"seed_{seed}"
            run_output.mkdir(parents=True, exist_ok=True)
            run_info = {
                "dataset": dataset,
                "seed": seed,
                "config": config,
                "checkpoints": {
                    "text": str(text_path),
                    "grounding": str(grounding_path),
                    "ura": str(ura_path),
                },
                "checkpoint_sha256": {
                    "text": sha256(text_path),
                    "grounding": sha256(grounding_path),
                    "ura": sha256(ura_path),
                },
                "splits": {},
            }
            for split in ("val", "test"):
                values = infer_scores(ts, model, loaders[split], device)
                cached_path = (
                    run_dir / "val_predictions.npz"
                    if split == "val"
                    else run_dir / "stage_predictions.npz"
                )
                with np.load(cached_path, allow_pickle=False) as cached:
                    if not np.array_equal(values["ids"], cached["ids"]):
                        raise AssertionError(f"ID mismatch: {dataset} {seed} {split}")
                    if not np.array_equal(values["labels"], cached["labels"]):
                        raise AssertionError(f"Label mismatch: {dataset} {seed} {split}")
                    cached_stages = cached["stage_logits"].copy()
                    absolute_difference = np.abs(values["stage_logits"] - cached_stages)
                    max_abs_diff = float(np.max(absolute_difference))
                    mean_abs_diff = float(np.mean(absolute_difference))
                    prediction_agreement = float(
                        np.mean(
                            values["stage_logits"].argmax(-1)
                            == cached_stages.argmax(-1)
                        )
                    )
                print(
                    f"  {split}: reconstruction max_diff={max_abs_diff:.6f}, "
                    f"mean_diff={mean_abs_diff:.6f}, "
                    f"prediction_agreement={prediction_agreement:.6f}",
                    flush=True,
                )
                if mean_abs_diff > 1e-1 or prediction_agreement < 0.995:
                    raise AssertionError(
                        f"Stage-logit mismatch {dataset} {seed} {split}: "
                        f"max={max_abs_diff}, agreement={prediction_agreement}"
                    )
                # Use the immutable cached specialist logits for all downstream
                # policy comparisons; only the learned route probabilities had
                # to be reconstructed from the checkpoint.
                values["stage_logits"] = cached_stages
                destination = run_output / f"{split}_router_scores.npz"
                np.savez_compressed(destination, **values)
                run_info["splits"][split] = {
                    "n": int(len(values["labels"])),
                    "max_cached_stage_logit_abs_diff": max_abs_diff,
                    "mean_cached_stage_logit_abs_diff": mean_abs_diff,
                    "cached_stage_prediction_agreement": prediction_agreement,
                    "output": str(destination),
                    "sha256": sha256(destination),
                }
            manifest["runs"].append(run_info)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    (output / "router_score_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved router scores to {output}", flush=True)


if __name__ == "__main__":
    main()
