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


def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _infer_arch_from_checkpoint(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Infer CLIP model constructor kwargs from Lightning checkpoint hyperparameters.
    We search typical locations ('hyper_parameters', 'hparams', plus nested dicts).
    Missing keys fall back to safe defaults consistent with training code.
    """
    hp = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}

    # Gather candidate dicts to scan
    cands: List[Mapping[str, Any]] = []
    if isinstance(hp, Mapping):
        cands.append(hp)
        for k in ("model", "clip", "clip_model", "cfg", "net"):
            v = hp.get(k)
            if isinstance(v, Mapping):
                cands.append(v)
            if k == "cfg" and isinstance(v, Mapping) and isinstance(v.get("model"), Mapping):
                cands.append(v["model"])  # cfg.model

    def pick(*keys, default=None):
        for d in cands:
            for k in keys:
                if k in d:
                    return d[k]
        return default

    # Reasonable defaults if not present (will be overwritten by checkpoint weights anyway)
    arch = {
        "input_sites": int(pick("input_sites", "input_genes", default=0) or 0),
        "embed_dim": int(pick("embed_dim", default=128) or 128),
        "img_enc_name": str(pick("img_enc_name", default="gigapath") or "gigapath"),
        "dnameth_enc_name": str(pick("dnameth_enc_name", "gene_enc_name", default="cpgpt") or "cpgpt"),
        "loss": str(pick("loss", default="CLIP") or "CLIP"),
        "img_finetune": bool(pick("img_finetune", default=False) or False),
        "dnameth_finetune": bool(pick("dnameth_finetune", "gene_finetune", default=False) or False),
        "n_tissue": pick("n_tissue", default=None),
        "n_region": pick("n_region", default=None),
        "image_size": int(pick("image_size", default=224) or 224),
        "temperature": float(pick("temperature", default=0.07) or 0.07),
        "world_size": int(pick("world_size", default=1) or 1),
        "rank": int(pick("rank", default=0) or 0),
        "img_proj": str(pick("img_proj", "image_proj", default="linear") or "linear"),
        "dnameth_proj": str(pick("dnameth_proj", "gene_proj", default="identity") or "identity"),
    }
    return arch


@dataclass(slots=True)
class ClipModelConfig:
    """Minimal config to materialise a CLIPModel from checkpoint only."""
    checkpoint_path: Path
    device: str | None = None
    batch_size: int = 4
    normalize_output: bool = True
    # Optional fusion controls for ClipFusionEmbeddingExtractor
    fusion: Mapping[str, Any] | None = None


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

        # Load checkpoint and infer architecture
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arch = _infer_arch_from_checkpoint(raw)

        clip_model = CLIPModel(
            input_sites=int(arch.get("input_sites", 0)),
            embed_dim=int(arch["embed_dim"]),
            img_enc_name=arch.get("img_enc_name", "gigapath"),
            dnameth_enc_name=arch.get("dnameth_enc_name", "cpgpt"),
            loss=arch.get("loss", "CLIP"),
            img_finetune=bool(arch.get("img_finetune", False)),
            dnameth_finetune=bool(arch.get("dnameth_finetune", False)),
            n_tissue=arch.get("n_tissue"),
            n_region=arch.get("n_region"),
            image_size=int(arch.get("image_size", 224)),
            temperature=float(arch.get("temperature", 0.07)),
            world_size=int(arch.get("world_size", 1)),
            rank=int(arch.get("rank", 0)),
            img_proj=arch.get("img_proj", "linear"),
            dnameth_proj=arch.get("dnameth_proj", "identity"),
        )

        state_dict = _extract_model_state(raw)
        incompatible = clip_model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning("Missing keys when loading %s: %s", ckpt_path, sorted(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys when loading %s: %s", ckpt_path, sorted(incompatible.unexpected_keys))

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
    """Extract only the DNA-methylation branch embeddings (CpGPT baseline)."""

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
                batch_dna = [sample.dnameth_path for sample in batch_samples]
                dna_embed = model.dnameth_encoder(batch_dna)
                if self.normalize_output:
                    dna_embed = F.normalize(dna_embed, p=2, dim=-1)
                for sample, vector in zip(batch_samples, dna_embed, strict=True):
                    results[sample.sample_id] = vector.detach().cpu().numpy()

        return results
