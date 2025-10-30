from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hescape.constants import DatasetEnum
from hescape.evaluation.data import EvaluationSample
from hescape.models import CLIPModel

logger = logging.getLogger(__name__)


def _default_device(explicit: str | None = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_model_state(ckpt: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Collect the actual model weights from a Lightning checkpoint."""
    if "state_dict" not in ckpt:
        return ckpt
    state_dict: Mapping[str, torch.Tensor] = ckpt["state_dict"]
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            filtered[key[len("model.") :]] = value
        else:
            filtered[key] = value
    return filtered


@dataclass(slots=True)
class ClipModelConfig:
    """Configuration required to materialise a CLIPModel from checkpoints."""

    checkpoint_path: Path
    model: Mapping[str, Any]
    image_encoder: Mapping[str, Any]
    dnameth_encoder: Mapping[str, Any]
    fusion: Mapping[str, Any] | None = None
    device: str | None = None
    batch_size: int = 4
    normalize_output: bool = True


class _ClipModelBundle:
    """Lazy loader wrapping the CLIP model instantiation and checkpoint restore."""

    def __init__(self, config: ClipModelConfig):
        self.cfg = config
        self.device = _default_device(config.device)
        self._model: CLIPModel | None = None

    def model(self) -> CLIPModel:
        if self._model is None:
            self._model = self._build()
        return self._model

    def _build(self) -> CLIPModel:
        ckpt_path = self.cfg.checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model_cfg = dict(self.cfg.model)
        img_cfg = dict(self.cfg.image_encoder)
        dna_cfg = dict(self.cfg.dnameth_encoder)

        clip_kwargs: Dict[str, Any] = {}
        clip_kwargs.update(img_cfg)
        clip_kwargs.update(dna_cfg)

        clip_model = CLIPModel(
            input_sites=int(model_cfg.get("input_sites", model_cfg.get("input_genes", 0))),
            embed_dim=int(model_cfg["embed_dim"]),
            img_enc_name=model_cfg.get("img_enc_name", "gigapath"),
            dnameth_enc_name=model_cfg.get("dnameth_enc_name", model_cfg.get("gene_enc_name", "cpgpt")),
            loss=model_cfg.get("loss", "CLIP"),
            img_finetune=bool(model_cfg.get("img_finetune", model_cfg.get("img_finetune", False))),
            dnameth_finetune=bool(model_cfg.get("dnameth_finetune", model_cfg.get("gene_finetune", False))),
            n_tissue=model_cfg.get("n_tissue"),
            n_region=model_cfg.get("n_region"),
            image_size=int(model_cfg.get("image_size", 224)),
            temperature=float(model_cfg.get("temperature", 0.07)),
            world_size=int(model_cfg.get("world_size", 1)),
            rank=int(model_cfg.get("rank", 0)),
            img_proj=model_cfg.get("img_proj", model_cfg.get("image_proj", "linear")),
            dnameth_proj=model_cfg.get("dnameth_proj", model_cfg.get("gene_proj", "identity")),
            **clip_kwargs,
        )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_model_state(state)
        incompatible = clip_model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning("Missing keys when loading %s: %s", ckpt_path, sorted(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.warning(
                "Unexpected keys when loading %s: %s", ckpt_path, sorted(incompatible.unexpected_keys)
            )

        clip_model.eval()
        clip_model.to(self.device)
        return clip_model


class ClipFusionEmbeddingExtractor:
    """Produces fused embeddings from both image and DNA methylation modalities."""

    def __init__(self, config: ClipModelConfig):
        self.bundle = _ClipModelBundle(config)
        fusion_cfg = config.fusion or {}
        self.modalities: Sequence[str] = fusion_cfg.get("modalities", ("image", "dnameth"))
        self.strategy: str = fusion_cfg.get("strategy", "mean")
        self.normalize_output: bool = config.normalize_output
        self.batch_size: int = max(1, config.batch_size)

    def embed(self, samples: Sequence[EvaluationSample]) -> Dict[str, np.ndarray]:
        model = self.bundle.model()
        device = self.bundle.device
        results: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch_samples = samples[start : start + self.batch_size]
                batch = {
                    DatasetEnum.IMG: [sample.image_path for sample in batch_samples],
                    DatasetEnum.DNAMETH: [sample.dnameth_path for sample in batch_samples],
                }
                image_embed, dnameth_embed, _ = model(batch, norm=True)
                fused = self._fuse(image_embed.to(device), dnameth_embed.to(device))
                if self.normalize_output:
                    fused = F.normalize(fused, p=2, dim=-1)
                for sample, vector in zip(batch_samples, fused, strict=True):
                    results[sample.sample_id] = vector.detach().cpu().numpy()

        return results

    def _fuse(self, image: torch.Tensor, dnameth: torch.Tensor) -> torch.Tensor:
        components: List[torch.Tensor] = []
        for modality in self.modalities:
            if modality == "image":
                components.append(image)
            elif modality in {"dnameth", "dna", "methylation"}:
                components.append(dnameth)
            else:
                raise ValueError(f"Unsupported modality {modality!r}")

        if not components:
            raise ValueError("No modalities selected for fusion.")

        if len(components) == 1:
            return components[0]

        if self.strategy == "mean":
            stacked = torch.stack(components, dim=0)
            return stacked.mean(dim=0)
        if self.strategy == "sum":
            stacked = torch.stack(components, dim=0)
            return stacked.sum(dim=0)
        if self.strategy == "concat":
            return torch.cat(components, dim=-1)

        raise ValueError(f"Unknown fusion strategy: {self.strategy!r}")


class ClipImageEmbeddingExtractor:
    """Extract only the image branch embeddings (e.g., baseline GigaPath features)."""

    def __init__(self, config: ClipModelConfig):
        self.bundle = _ClipModelBundle(config)
        self.normalize_output: bool = config.normalize_output
        self.batch_size: int = max(1, config.batch_size)

    def embed(self, samples: Sequence[EvaluationSample]) -> Dict[str, np.ndarray]:
        model = self.bundle.model()
        results: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch_samples = samples[start : start + self.batch_size]
                batch_images = [sample.image_path for sample in batch_samples]
                image_embed = model.image_encoder(batch_images)
                if self.normalize_output:
                    image_embed = F.normalize(image_embed, p=2, dim=-1)
                for sample, vector in zip(batch_samples, image_embed, strict=True):
                    results[sample.sample_id] = vector.detach().cpu().numpy()

        return results


class ClipDnaEmbeddingExtractor:
    """Extract only the DNA methylation branch embeddings (CpGPT baseline)."""

    def __init__(self, config: ClipModelConfig):
        self.bundle = _ClipModelBundle(config)
        self.normalize_output: bool = config.normalize_output
        self.batch_size: int = max(1, config.batch_size)

    def embed(self, samples: Sequence[EvaluationSample]) -> Dict[str, np.ndarray]:
        model = self.bundle.model()
        results: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch_samples = samples[start : start + self.batch_size]
                batch_beta = [sample.dnameth_path for sample in batch_samples]
                dnameth_embed = model.dnameth_encoder(batch_beta)
                if self.normalize_output:
                    dnameth_embed = F.normalize(dnameth_embed, p=2, dim=-1)
                for sample, vector in zip(batch_samples, dnameth_embed, strict=True):
                    results[sample.sample_id] = vector.detach().cpu().numpy()

        return results
