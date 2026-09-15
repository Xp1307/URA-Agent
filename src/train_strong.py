"""End-to-end multimodal fake-news baselines and uncertainty-routed agent.

The implementation keeps the data split fixed across methods and supports:
text_ft, concat_ft, cafe_adapted, mcot, and ura_agent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


DATASETS = {
    "weibo": {
        "text_model": "bert-base-chinese",
        "vision_model": "google/vit-base-patch16-224-in21k",
        "image_dir": "weibo_images/images_all",
        "max_length": 150,
        "joint_max_length": 256,
    },
    "pheme": {
        "text_model": "roberta-base",
        "vision_model": "google/vit-base-patch16-224-in21k",
        "image_dir": "pheme_images/images_all",
        "max_length": 80,
        "joint_max_length": 160,
    },
}
DESCRIPTION_KEY = "Qwen3-VL-8B-Instruct_generation"
IMAGE_METHODS = {"concat_ft", "cafe_adapted", "mcot", "ura_agent"}
JOINT_METHODS = {"joint_ft", "joint_refine", "ura_agent"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(-1).clamp_min(1e-8)
    return -(p * p.log()).sum(-1, keepdim=True) / math.log(p.shape[-1])


class NewsDataset(Dataset):
    def __init__(self, project_root: Path, dataset: str, split: str):
        cfg = DATASETS[dataset]
        base = project_root / "datasets" / dataset
        json_path = base / "data_json" / f"{split}_imdes.json"
        if not json_path.exists():
            json_path = base / "data_json" / f"{split}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.records = list(payload.values()) if isinstance(payload, dict) else payload

        image_root = base / cfg["image_dir"]
        image_index: dict[str, Path] = {}
        for path in image_root.rglob("*"):
            if path.is_file():
                image_index[path.name] = path
                image_index[path.stem] = path
        self.items: list[dict[str, Any]] = []
        missing: list[str] = []
        for rec in self.records:
            img_id = str(rec["img_id"])
            image_path = image_index.get(Path(img_id).name) or image_index.get(Path(img_id).stem)
            if image_path is None:
                missing.append(img_id)
                continue
            self.items.append(
                {
                    "id": str(rec.get("tweet_id", len(self.items))),
                    "text": str(rec.get("tweet_content", "")),
                    "description": str(rec.get(DESCRIPTION_KEY, "")),
                    "image_path": str(image_path),
                    "label": int(rec["tweet_label"]),
                }
            )
        if missing:
            raise FileNotFoundError(f"{len(missing)} images are missing; first={missing[0]}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class BatchCollator:
    def __init__(self, dataset: str, method: str):
        cfg = DATASETS[dataset]
        self.method = method
        self.max_length = cfg["max_length"]
        self.joint_max_length = cfg["joint_max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"], local_files_only=True)
        self.image_processor = None
        if method in IMAGE_METHODS:
            self.image_processor = AutoImageProcessor.from_pretrained(
                cfg["vision_model"], local_files_only=True
            )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        text = self.tokenizer(
            [r["text"] for r in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            "ids": [r["id"] for r in rows],
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        }
        if self.method in JOINT_METHODS:
            joint = self.tokenizer(
                [r["text"] for r in rows],
                [r["description"] for r in rows],
                padding=True,
                truncation=True,
                max_length=self.joint_max_length,
                return_tensors="pt",
            )
            batch["joint_input_ids"] = joint["input_ids"]
            batch["joint_attention_mask"] = joint["attention_mask"]
        if self.method in IMAGE_METHODS:
            images = []
            for row in rows:
                with Image.open(row["image_path"]) as image:
                    images.append(image.convert("RGB"))
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
        return batch


def _freeze_and_open_last_layers(model: nn.Module, n_layers: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError(f"Cannot locate transformer layers for {model.__class__.__name__}")
    if n_layers > 0:
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    for name in ("layernorm", "pooler"):
        module = getattr(model, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


class Backbone(nn.Module):
    def __init__(
        self,
        dataset: str,
        need_image: bool,
        text_layers: int,
        vision_layers: int,
        pooling: str = "cls",
    ):
        super().__init__()
        cfg = DATASETS[dataset]
        self.text = AutoModel.from_pretrained(cfg["text_model"], local_files_only=True)
        _freeze_and_open_last_layers(self.text, text_layers)
        self.text_dim = int(self.text.config.hidden_size)
        self.pooling = pooling
        self.vision = None
        self.vision_dim = 0
        if need_image:
            self.vision = AutoModel.from_pretrained(cfg["vision_model"], local_files_only=True)
            _freeze_and_open_last_layers(self.vision, vision_layers)
            self.vision_dim = int(self.vision.config.hidden_size)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        sequence = out.last_hidden_state
        pooled = sequence[:, 0]
        if self.pooling == "cls_mean":
            mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
            mean = (sequence * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = 0.5 * (pooled + mean)
        return sequence, pooled

    def encode_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vision is None:
            raise RuntimeError("Vision backbone was not initialized")
        out = self.vision(pixel_values=pixel_values)
        return out.last_hidden_state, out.last_hidden_state[:, 0]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextFT(nn.Module):
    def __init__(self, dataset: str, text_layers: int, pooling: str = "cls", **_: Any):
        super().__init__()
        self.backbone = Backbone(dataset, False, text_layers, 0, pooling)
        self.classifier = MLP(self.backbone.text_dim, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, text = self.backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        return {"logits": self.classifier(text), "aux_loss": text.new_zeros(())}


class JointFT(nn.Module):
    """Cross-encode post text and image caption as one grounded sequence."""

    def __init__(self, dataset: str, text_layers: int, pooling: str = "cls", **_: Any):
        super().__init__()
        self.backbone = Backbone(dataset, False, text_layers, 0, pooling)
        self.classifier = MLP(self.backbone.text_dim, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, joint = self.backbone.encode_text(
            batch["joint_input_ids"], batch["joint_attention_mask"]
        )
        return {"logits": self.classifier(joint), "aux_loss": joint.new_zeros(())}


class ConcatFT(nn.Module):
    def __init__(self, dataset: str, text_layers: int, vision_layers: int, dim: int = 256, pooling: str = "cls", **_: Any):
        super().__init__()
        self.backbone = Backbone(dataset, True, text_layers, vision_layers, pooling)
        self.text_proj = MLP(self.backbone.text_dim, dim, hidden=dim)
        self.image_proj = MLP(self.backbone.vision_dim, dim, hidden=dim)
        self.classifier = MLP(2 * dim, 2, hidden=dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, text = self.backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        _, image = self.backbone.encode_image(batch["pixel_values"])
        text, image = self.text_proj(text), self.image_proj(image)
        return {"logits": self.classifier(torch.cat([text, image], -1)), "aux_loss": text.new_zeros(())}


class CAFEAdapted(nn.Module):
    """CAFE-style ambiguity-aware fusion with modern BERT/ViT encoders."""

    def __init__(self, dataset: str, text_layers: int, vision_layers: int, dim: int = 256, pooling: str = "cls", **_: Any):
        super().__init__()
        self.backbone = Backbone(dataset, True, text_layers, vision_layers, pooling)
        self.text_proj = MLP(self.backbone.text_dim, dim, hidden=dim)
        self.image_proj = MLP(self.backbone.vision_dim, dim, hidden=dim)
        self.text_gauss = nn.Linear(dim, 64)
        self.image_gauss = nn.Linear(dim, 64)
        self.text_uni = MLP(dim, 64, hidden=128)
        self.image_uni = MLP(dim, 64, hidden=128)
        self.correlation = MLP(2 * dim, 64, hidden=128)
        self.classifier = MLP(64 * 3, 2, hidden=128)

    @staticmethod
    def symmetric_kl(mu1: torch.Tensor, lv1: torch.Tensor, mu2: torch.Tensor, lv2: torch.Tensor) -> torch.Tensor:
        v1, v2 = lv1.exp(), lv2.exp()
        delta = (mu1 - mu2).square()
        kl12 = 0.5 * (lv2 - lv1 + (v1 + delta) / v2 - 1).mean(-1, keepdim=True)
        kl21 = 0.5 * (lv1 - lv2 + (v2 + delta) / v1 - 1).mean(-1, keepdim=True)
        return 0.5 * (kl12 + kl21)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, text = self.backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        _, image = self.backbone.encode_image(batch["pixel_values"])
        text, image = self.text_proj(text), self.image_proj(image)
        mt, lvt = self.text_gauss(text).chunk(2, -1)
        mi, lvi = self.image_gauss(image).chunk(2, -1)
        lvt, lvi = lvt.clamp(-5, 5), lvi.clamp(-5, 5)
        skl = self.symmetric_kl(mt, lvt, mi, lvi)
        w_corr = torch.sigmoid(skl - 1.0)
        uni_t, uni_i = self.text_uni(text), self.image_uni(image)
        corr = self.correlation(torch.cat([text * image, (text - image).abs()], -1))
        fused = torch.cat([(1 - w_corr) * uni_t, (1 - w_corr) * uni_i, w_corr * corr], -1)

        matched = F.cosine_similarity(text, image).mean()
        unmatched = F.cosine_similarity(text, image.roll(1, 0)).mean()
        alignment = (1 - matched) + F.relu(unmatched - 0.2)
        return {"logits": self.classifier(fused), "aux_loss": 0.1 * alignment}


def symmetric_infonce(text: torch.Tensor, image: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    text, image = F.normalize(text, dim=-1), F.normalize(image, dim=-1)
    scores = text @ image.T / temperature
    target = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target))


def sinkhorn_loss(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    image_tokens: torch.Tensor,
    epsilon: float = 0.1,
    iterations: int = 5,
) -> torch.Tensor:
    """Entropic OT cost between valid text tokens and a 7x7 visual patch grid."""
    text_tokens = F.normalize(text_tokens.float(), dim=-1)
    image_tokens = F.normalize(image_tokens.float(), dim=-1)
    cost = (1 - torch.einsum("btd,bvd->btv", text_tokens, image_tokens)).clamp(0, 2)
    valid = text_mask.bool()
    a = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
    log_a = torch.where(valid, a.clamp_min(1e-12).log(), torch.full_like(a, -1e4))
    log_b = torch.full(
        (image_tokens.shape[0], image_tokens.shape[1]),
        -math.log(image_tokens.shape[1]),
        device=image_tokens.device,
    )
    log_kernel = -cost / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
    log_plan = log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(1)
    plan = log_plan.exp() * valid.unsqueeze(-1)
    return (plan * cost).sum((-1, -2)).mean()


class MCOT(nn.Module):
    """Cross-attention, contrastive alignment, and entropic optimal transport."""

    def __init__(self, dataset: str, text_layers: int, vision_layers: int, dim: int = 256, pooling: str = "cls", **_: Any):
        super().__init__()
        self.backbone = Backbone(dataset, True, text_layers, vision_layers, pooling)
        self.text_token_proj = nn.Linear(self.backbone.text_dim, dim)
        self.image_token_proj = nn.Linear(self.backbone.vision_dim, dim)
        self.text_to_image = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.classifier = MLP(2 * dim, 2, hidden=dim)
        self.dataset = dataset

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_seq, _ = self.backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        image_seq, _ = self.backbone.encode_image(batch["pixel_values"])
        text_seq = self.text_token_proj(text_seq)
        image_seq = self.image_token_proj(image_seq)
        text_pad = ~batch["attention_mask"].bool()
        t_cross, _ = self.text_to_image(text_seq, image_seq, image_seq, need_weights=False)
        i_cross, _ = self.image_to_text(
            image_seq, text_seq, text_seq, key_padding_mask=text_pad, need_weights=False
        )
        text_seq = self.norm_t(text_seq + t_cross)
        image_seq = self.norm_i(image_seq + i_cross)
        text_pool, image_pool = text_seq[:, 0], image_seq[:, 0]

        contrast = symmetric_infonce(text_pool, image_pool)
        patches = image_seq[:, 1:]
        side = int(math.sqrt(patches.shape[1]))
        patches = patches[:, : side * side].reshape(-1, side, side, patches.shape[-1])
        patches = F.adaptive_avg_pool2d(patches.permute(0, 3, 1, 2), (7, 7))
        patches = patches.flatten(2).transpose(1, 2)
        ot = sinkhorn_loss(text_seq[:, 1:], batch["attention_mask"][:, 1:], patches)
        beta, gamma = (0.4, 0.4) if self.dataset == "weibo" else (0.1, 0.1)
        return {
            "logits": self.classifier(torch.cat([text_pool, image_pool], -1)),
            "aux_loss": beta * contrast + gamma * ot,
            "contrastive_loss": contrast.detach(),
            "ot_loss": ot.detach(),
        }


class URAAgent(nn.Module):
    """Sequential uncertainty-routed text, vision, grounding, consistency agents."""

    def __init__(
        self,
        dataset: str,
        text_layers: int,
        vision_layers: int,
        dim: int = 256,
        route_cost: float = 0.003,
        pooling: str = "cls",
        **_: Any,
    ):
        super().__init__()
        self.backbone = Backbone(dataset, True, text_layers, vision_layers, pooling)
        grounding_cfg = DATASETS[dataset]
        self.grounding_text = AutoModel.from_pretrained(
            grounding_cfg["text_model"], local_files_only=True
        )
        _freeze_and_open_last_layers(self.grounding_text, text_layers)
        self.text_proj = MLP(self.backbone.text_dim, dim, hidden=dim)
        self.image_proj = MLP(self.backbone.vision_dim, dim, hidden=dim)
        self.grounding_proj = MLP(self.backbone.text_dim, dim, hidden=dim)
        pair_dim = dim * 4
        # Keep the text agent architecture identical to TextFT so that its
        # validated checkpoint can initialize the sequential system.
        self.text_head = MLP(self.backbone.text_dim, 2, hidden=dim)
        self.image_head = MLP(pair_dim, 2, hidden=dim)
        self.grounding_head = MLP(self.backbone.text_dim, 2, hidden=dim)
        self.consistency_head = MLP(dim * 6, 2, hidden=dim)
        self.router_image = MLP(dim + 1, 1, hidden=128)
        self.router_desc = MLP(dim + 3, 1, hidden=128)
        self.router_consistency = MLP(dim + 5, 1, hidden=128)
        self.route_cost = route_cost

    @staticmethod
    def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, (a - b).abs(), a * b], -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, text_raw = self.backbone.encode_text(batch["input_ids"], batch["attention_mask"])
        _, image_raw = self.backbone.encode_image(batch["pixel_values"])
        grounding_out = self.grounding_text(
            input_ids=batch["joint_input_ids"],
            attention_mask=batch["joint_attention_mask"],
        )
        grounding_sequence = grounding_out.last_hidden_state
        grounding_raw = grounding_sequence[:, 0]
        if self.backbone.pooling == "cls_mean":
            mask = batch["joint_attention_mask"].unsqueeze(-1).to(grounding_sequence.dtype)
            grounding_mean = (grounding_sequence * mask).sum(1) / mask.sum(1).clamp_min(1)
            grounding_raw = 0.5 * (grounding_raw + grounding_mean)
        text = self.text_proj(text_raw)
        image = self.image_proj(image_raw)
        grounding = self.grounding_proj(grounding_raw)

        stage0 = self.text_head(text_raw)
        ent0 = entropy_from_logits(stage0)
        route_logit1 = self.router_image(torch.cat([text, ent0], -1))
        route_prob1 = torch.sigmoid(route_logit1)
        gate1 = route_prob1 * ent0.detach()
        image_logits = self.image_head(self.pair_features(text, image))
        stage1 = (1 - gate1) * stage0 + gate1 * image_logits

        ent1 = entropy_from_logits(stage1)
        stats1 = torch.cat([ent0, ent1, F.cosine_similarity(text, image).unsqueeze(-1)], -1)
        route_logit2 = self.router_desc(torch.cat([text, stats1], -1))
        route_prob2 = torch.sigmoid(route_logit2)
        gate2 = route_prob2 * ent1.detach()
        grounding_logits = self.grounding_head(grounding_raw)
        stage2 = (1 - gate2) * stage1 + gate2 * grounding_logits

        ent2 = entropy_from_logits(stage2)
        sim_ti = F.cosine_similarity(text, image).unsqueeze(-1)
        sim_td = F.cosine_similarity(text, grounding).unsqueeze(-1)
        router3_stats = torch.cat([ent0, ent1, ent2, sim_ti, sim_td], -1)
        route_logit3 = self.router_consistency(torch.cat([text, router3_stats], -1))
        route_prob3 = torch.sigmoid(route_logit3)
        gate3 = route_prob3 * ent2.detach()
        consistency = torch.cat(
            [
                text * image,
                text * grounding,
                image * grounding,
                (text - image).abs(),
                (text - grounding).abs(),
                (image - grounding).abs(),
            ],
            -1,
        )
        consistency_logits = self.consistency_head(consistency)
        stage3 = (1 - gate3) * stage2 + gate3 * consistency_logits
        gates = torch.cat([gate1, gate2, gate3], -1)
        route_penalty = self.route_cost * gates.mean()
        return {
            "logits": stage3,
            # Routing operates over the actual specialist decisions.  The
            # differentiable stage3 path above remains the training objective,
            # while validation selects how far an uncertain sample should go.
            "stage_logits": torch.stack(
                [stage0, image_logits, grounding_logits, consistency_logits], 1
            ),
            "expert_logits": torch.stack([image_logits, grounding_logits, consistency_logits], 1),
            "route_probs": torch.cat([route_prob1, route_prob2, route_prob3], -1),
            "route_logits": torch.cat([route_logit1, route_logit2, route_logit3], -1),
            "gates": gates,
            "aux_loss": route_penalty,
        }


MODEL_CLASSES = {
    "text_ft": TextFT,
    "text_refine": TextFT,
    "joint_ft": JointFT,
    "joint_refine": JointFT,
    "concat_ft": ConcatFT,
    "cafe_adapted": CAFEAdapted,
    "mcot": MCOT,
    "ura_agent": URAAgent,
}


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == labels[mask]).mean() - conf[mask].mean())
    return float(ece)


def metric_dict(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = torch.from_numpy(logits).softmax(-1).numpy()
    pred = probs.argmax(1)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, pred, labels=[0, 1], zero_division=0
    )
    try:
        auroc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1_per_class[1]),
        "real_f1": float(f1_per_class[0]),
        "auroc": float(auroc),
        "ece": calibration_error(labels, probs),
    }


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_ids, all_labels, all_logits, all_stages = [], [], [], []
    for batch in loader:
        ids = batch.pop("ids")
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        all_ids.extend(ids)
        all_labels.append(batch["labels"].cpu())
        all_logits.append(out["logits"].float().cpu())
        if "stage_logits" in out:
            all_stages.append(out["stage_logits"].float().cpu())
    result = {
        "ids": all_ids,
        "labels": torch.cat(all_labels).numpy(),
        "logits": torch.cat(all_logits).numpy(),
    }
    if all_stages:
        result["stage_logits"] = torch.cat(all_stages).numpy()
    return result


def route_agent(
    stage_logits: np.ndarray, thresholds: float | list[float] | tuple[float, ...]
) -> tuple[np.ndarray, float]:
    stages = torch.from_numpy(stage_logits)
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)] * (stages.shape[1] - 1)
    thresholds = list(thresholds)
    if len(thresholds) != stages.shape[1] - 1:
        raise ValueError(
            f"Expected {stages.shape[1] - 1} routing thresholds, got {len(thresholds)}"
        )
    current = stages[:, 0].clone()
    active = current.softmax(-1).max(-1).values < thresholds[0]
    calls = torch.zeros(stages.shape[0])
    for stage in range(1, stages.shape[1]):
        current[active] = stages[active, stage]
        calls[active] += 1
        if stage < stages.shape[1] - 1:
            still_uncertain = current.softmax(-1).max(-1).values < thresholds[stage]
            active = active & still_uncertain
    return current.numpy(), float(calls.mean())


def tune_agent_threshold(
    labels: np.ndarray, stage_logits: np.ndarray, selection_metric: str = "macro_f1"
) -> tuple[list[float], dict[str, float]]:
    # Each expert has different reliability.  Tune a compact, validation-only
    # policy rather than forcing one confidence threshold on every stage.
    # 0.50 means "stop", while 1.01 is the explicit "always call" endpoint.
    # A coarse grid limits validation overfitting on these small datasets.
    grid = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.01])
    best: tuple[float, float, tuple[float, ...], dict[str, float]] | None = None
    for thresholds in itertools.product(grid, repeat=stage_logits.shape[1] - 1):
        logits, calls = route_agent(stage_logits, thresholds)
        metrics = metric_dict(labels, logits)
        candidate = (metrics[selection_metric], -calls, tuple(map(float, thresholds)), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    metrics = best[3]
    metrics["avg_optional_calls"] = -best[1]
    return list(best[2]), metrics


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable or not key.startswith(("backbone.", "grounding_text."))
    }


@dataclass
class RunConfig:
    dataset: str
    method: str
    seed: int
    epochs: int
    batch_size: int
    encoder_lr: float
    head_lr: float
    weight_decay: float
    text_layers: int
    vision_layers: int
    pooling: str
    class_weighting: str
    selection_metric: str
    merge_val: bool
    agent_text_source: str
    patience: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=MODEL_CLASSES, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--vision-layers", type=int, default=2)
    parser.add_argument("--pooling", choices=("cls", "cls_mean"), default="cls")
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument(
        "--merge-val",
        action="store_true",
        help="Final-fit mode: train on train+val for exactly --epochs and keep the last epoch.",
    )
    parser.add_argument(
        "--agent-text-source",
        choices=("auto", "text_ft", "text_refine"),
        default="auto",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/strong_results"))
    parser.add_argument(
        "--init-results-dir",
        type=Path,
        default=Path("artifacts/strong_results"),
        help="Fallback results root for pretrained text and grounding experts.",
    )
    args = parser.parse_args()

    cfg = RunConfig(
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        text_layers=args.text_layers,
        vision_layers=args.vision_layers,
        pooling=args.pooling,
        class_weighting=args.class_weighting,
        selection_metric=args.selection_metric,
        merge_val=args.merge_val,
        agent_text_source=args.agent_text_source,
        patience=args.patience,
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    output = args.project_root / args.output_dir / args.dataset / args.method / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    datasets = {s: NewsDataset(args.project_root, args.dataset, s) for s in ("train", "val", "test")}
    if args.merge_val:
        datasets["train"].items.extend(datasets["val"].items)
    collator = BatchCollator(args.dataset, args.method)
    loaders = {
        split: DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collator,
        )
        for split, data in datasets.items()
    }
    model = MODEL_CLASSES[args.method](
        dataset=args.dataset,
        text_layers=args.text_layers,
        vision_layers=args.vision_layers,
        pooling=args.pooling,
    ).to(device)
    initialized_from = None
    initialized_grounding_from = None
    output_root = args.project_root / args.output_dir
    fallback_root = args.project_root / args.init_results_dir

    def find_checkpoint(method: str) -> Path | None:
        for result_root in dict.fromkeys((output_root, fallback_root)):
            candidate = (
                result_root
                / args.dataset
                / method
                / f"seed_{args.seed}"
                / "best_trainable.pt"
            )
            if candidate.exists():
                return candidate
        return None

    if args.method in {"text_refine", "joint_refine"}:
        source_method = "text_ft" if args.method == "text_refine" else "joint_ft"
        source_checkpoint = find_checkpoint(source_method)
        if source_checkpoint is not None:
            source_state = torch.load(
                source_checkpoint, map_location="cpu", weights_only=True
            )["state_dict"]
            model.load_state_dict(source_state, strict=False)
            initialized_from = str(source_checkpoint)
            print(f"Initialized refinement from {source_checkpoint}", flush=True)
    elif args.method == "ura_agent":
        if args.agent_text_source == "auto":
            text_checkpoint = find_checkpoint("text_refine") or find_checkpoint("text_ft")
        else:
            text_checkpoint = find_checkpoint(args.agent_text_source)
        if text_checkpoint is not None:
            text_state = torch.load(
                text_checkpoint, map_location="cpu", weights_only=True
            )["state_dict"]
            mapped = {}
            for key, value in text_state.items():
                if key.startswith("backbone.text."):
                    mapped[key] = value
                elif key.startswith("classifier."):
                    mapped["text_head." + key.removeprefix("classifier.")] = value
            model.load_state_dict(mapped, strict=False)
            initialized_from = str(text_checkpoint)
            for param in model.backbone.text.parameters():
                param.requires_grad = False
            for param in model.text_head.parameters():
                param.requires_grad = False
            print(f"Initialized and froze Text Agent from {text_checkpoint}", flush=True)

        grounding_checkpoint = find_checkpoint("joint_refine") or find_checkpoint("joint_ft")
        if grounding_checkpoint is not None:
            grounding_state = torch.load(
                grounding_checkpoint, map_location="cpu", weights_only=True
            )["state_dict"]
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
            initialized_grounding_from = str(grounding_checkpoint)
            for param in model.grounding_text.parameters():
                param.requires_grad = False
            for param in model.grounding_head.parameters():
                param.requires_grad = False
            print(
                f"Initialized and froze Grounding Agent from {grounding_checkpoint}",
                flush=True,
            )
        print("Training visual/consistency experts and controller", flush=True)
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if name.startswith(("backbone.", "grounding_text.")) else head_params).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(loaders["train"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )
    labels_train = np.array([x["label"] for x in datasets["train"].items])
    counts = np.bincount(labels_train, minlength=2)
    weights = len(labels_train) / (2 * np.maximum(counts, 1))
    class_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if args.class_weighting == "balanced"
        else None
    )

    best_score, stale = -1.0, 0
    checkpoint = output / "best_trainable.pt"
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loaders["train"]:
            batch.pop("ids")
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch)
                loss = F.cross_entropy(out["logits"], batch["labels"], weight=class_weights)
                loss = loss + out.get("aux_loss", 0.0)
                if args.method == "ura_agent":
                    stages = out["stage_logits"]
                    experts = out["expert_logits"]
                    loss = loss + 0.60 * F.cross_entropy(stages[:, 0], batch["labels"], weight=class_weights)
                    loss = loss + 0.10 * F.cross_entropy(stages[:, 1], batch["labels"], weight=class_weights)
                    loss = loss + 0.10 * F.cross_entropy(stages[:, 2], batch["labels"], weight=class_weights)
                    loss = loss + 0.15 * F.cross_entropy(experts[:, 0], batch["labels"], weight=class_weights)
                    loss = loss + 0.15 * F.cross_entropy(experts[:, 1], batch["labels"], weight=class_weights)
                    loss = loss + 0.10 * F.cross_entropy(experts[:, 2], batch["labels"], weight=class_weights)
                    with torch.no_grad():
                        base_ce = F.cross_entropy(stages[:, 0], batch["labels"], reduction="none")
                        stage1_ce = F.cross_entropy(stages[:, 1], batch["labels"], reduction="none")
                        stage2_ce = F.cross_entropy(stages[:, 2], batch["labels"], reduction="none")
                        expert_ce = torch.stack(
                            [F.cross_entropy(experts[:, i], batch["labels"], reduction="none") for i in range(3)],
                            1,
                        )
                        route_targets = torch.stack(
                            [expert_ce[:, 0] < base_ce, expert_ce[:, 1] < stage1_ce, expert_ce[:, 2] < stage2_ce],
                            1,
                        ).float()
                    loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                        out["route_logits"], route_targets
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * batch["labels"].shape[0]
            seen += batch["labels"].shape[0]

        val = infer(model, loaders["val"], device)
        if args.method == "ura_agent":
            threshold, val_metrics = tune_agent_threshold(
                val["labels"], val["stage_logits"], args.selection_metric
            )
        else:
            threshold = None
            val_metrics = metric_dict(val["labels"], val["logits"])
        row = {"epoch": epoch, "train_loss": running / seen, **val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = val_metrics[args.selection_metric]
        improved = score > best_score + 1e-5
        should_save = epoch == args.epochs if args.merge_val else improved
        if should_save:
            best_score, stale = score, 0
            torch.save(
                {"state_dict": trainable_state_dict(model), "threshold": threshold, "epoch": epoch},
                checkpoint,
            )
        elif not args.merge_val:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=False)
    val = infer(model, loaders["val"], device)
    test = infer(model, loaders["test"], device)
    if args.method == "ura_agent":
        saved_threshold = saved["threshold"]
        threshold = (
            [float(x) for x in saved_threshold]
            if isinstance(saved_threshold, (list, tuple))
            else float(saved_threshold)
        )
        val_logits, val_calls = route_agent(val["stage_logits"], threshold)
        test_logits, test_calls = route_agent(test["stage_logits"], threshold)
        val_metrics, test_metrics = metric_dict(val["labels"], val_logits), metric_dict(test["labels"], test_logits)
        val_metrics["avg_optional_calls"] = val_calls
        test_metrics["avg_optional_calls"] = test_calls
    else:
        threshold = None
        val_logits, test_logits = val["logits"], test["logits"]
        val_metrics, test_metrics = metric_dict(val["labels"], val_logits), metric_dict(test["labels"], test_logits)

    result = {
        "config": asdict(cfg),
        "best_epoch": int(saved["epoch"]),
        "routing_threshold": threshold,
        "runtime_seconds": time.time() - start,
        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_total": sum(p.numel() for p in model.parameters()),
        "initialized_from": initialized_from,
        "initialized_grounding_from": initialized_grounding_from,
        "val": val_metrics,
        "test": test_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz",
        ids=np.asarray(test["ids"]),
        labels=test["labels"],
        logits=test_logits,
    )
    np.savez_compressed(
        output / "val_predictions.npz",
        ids=np.asarray(val["ids"]),
        labels=val["labels"],
        logits=val_logits,
        **({"stage_logits": val["stage_logits"]} if "stage_logits" in val else {}),
    )
    if "stage_logits" in test:
        # Keep the routed logits in predictions.npz for backward compatibility,
        # and persist all stages separately for routing-policy ablations.
        np.savez_compressed(
            output / "stage_predictions.npz",
            ids=np.asarray(test["ids"]),
            labels=test["labels"],
            stage_logits=test["stage_logits"],
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
