from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from hescape.constants import DatasetEnum
from hescape.evaluation.data import EvaluationSample
from hescape.models import CLIPModel

# New imports for base encoders & caching
from hescape.models.image_models._gigapath import _build_gigapath_model
from gigapath import pipeline
from hescape.models._cache import EmbeddingCache
from hescape.models.dnameth_models._cpgpt import CpGPTRunner

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
        return ckpt  # raw state_dict
    state_dict: Mapping[str, torch.Tensor] = ckpt["state_dict"]
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            filtered[key[len("model.") :]] = value
        else:
            filtered[key] = value
    return filtered


def _load_arch_from_hparams(path: Path) -> Dict[str, Any]:
    """
    Load all model hyperparameters from a Lightning CSVLogger hparams.yaml.

    This is strict: we index mandatory keys directly so a KeyError is raised if anything is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"hparams.yaml not found: {path}")

    hp = OmegaConf.load(str(path))
    hp = OmegaConf.to_container(hp, resolve=True)  # -> plain dict

    # Required sections
    lit = hp["model"]["litmodule"]  # raises KeyError if missing
    pretrain = hp["paths"]["pretrain_weights"]  # raises KeyError if missing

    # Map training-time names -> CLIPModel constructor kwargs
    arch = {
        # encoder names and dims
        "input_sites": int(lit["input_genes"]),
        "embed_dim": int(lit["embed_dim"]),
        "img_enc_name": str(lit["img_enc_name"]),
        "dnameth_enc_name": str(lit["gene_enc_name"]),
        "loss": str(lit["loss"]),

        # projections
        "img_proj": str(lit["img_proj"]),
        "dnameth_proj": str(lit["gene_proj"]),

        # finetuning switches
        "tile_finetune": bool(lit["tile_finetune"]),
        "slide_finetune": bool(lit["slide_finetune"]),
        "dnameth_finetune": bool(lit["gene_finetune"]),

        # optional-but-present in your YAML (allowed to be None)
        "n_tissue": lit.get("n_tissue", None),
        "n_region": lit.get("n_region", None),
        "image_size": int(lit["image_size"]),
        "temperature": float(lit["temperature"]),

        # encoder resource roots used at train time
        "img_enc_path": str(pretrain["img_enc_path"]),
        "dnameth_enc_path": str(pretrain["gene_enc_path"]),
    }
    return arch


@dataclass(slots=True)
class ClipModelConfig:
    """Minimal config to materialise a CLIPModel strictly from YAML + weights."""
    checkpoint_path: Path
    hparams_path: Path
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
            self._build_and_cache()
        return self._model  # type: ignore[return-value]

    def _build_and_cache(self) -> None:
        ckpt_path = self.cfg.checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # 1) Strictly parse architecture from user-specified hparams.yaml
        arch = _load_arch_from_hparams(self.cfg.hparams_path)

        # 2) Instantiate model with exactly those parameters (no assumptions)
        clip_model = CLIPModel(
            input_sites=arch["input_sites"],
            embed_dim=arch["embed_dim"],
            img_enc_name=arch["img_enc_name"],
            dnameth_enc_name=arch["dnameth_enc_name"],
            loss=arch["loss"],
            tile_finetune=arch["tile_finetune"],
            slide_finetune=arch["slide_finetune"],
            dnameth_finetune=arch["dnameth_finetune"],
            n_tissue=arch["n_tissue"],
            n_region=arch["n_region"],
            image_size=arch["image_size"],
            temperature=arch["temperature"],
            img_proj=arch["img_proj"],
            dnameth_proj=arch["dnameth_proj"],
            img_enc_path=arch["img_enc_path"],
            dnameth_enc_path=arch["dnameth_enc_path"],
        )

        # 3) Load weights
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_model_state(state)
        incompatible = clip_model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning("Missing keys when loading %s: %s", ckpt_path, sorted(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys when loading %s: %s", ckpt_path, sorted(incompatible.unexpected_keys))

        clip_model.eval()
        clip_model.to(self.device)
        self._model = clip_model


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
    """Extract only the image branch embeddings (e.g., baseline GigaPath features with CLIP head)."""

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
    """Extract only the DNA-methylation branch embeddings (CpGPT baseline through CLIP head)."""

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


class GigaPathBaseEmbeddingExtractor:
    """
    Extract raw slide embeddings directly from the GigaPath slide encoder (no CLIP projection head).
    Uses the same caching policy for tile embeddings as the ImageEncoder.
    """

    def __init__(self, config: ClipModelConfig):
        self.device = _default_device(config.device)
        # Load pre-trained tile/slide encoders from the GigaPath pipeline
        self.tile_encoder, self.slide_encoder = _build_gigapath_model()
        self.normalize_output: bool = config.normalize_output
        self.batch_size: int = max(1, config.batch_size)
        self.tile_cache = EmbeddingCache("tile_embeddings")

    def _tile_outputs_for_slide(self, slide_dir: str) -> Mapping[str, Any]:
        def _compute():
            image_paths = [
                os.path.join(slide_dir, fname)
                for fname in os.listdir(slide_dir)
                if fname.lower().endswith(".png")
            ]
            with torch.no_grad():
                return pipeline.run_inference_with_tile_encoder(
                    image_paths, self.tile_encoder, batch_size=64
                )
        return self.tile_cache.get_or_compute(slide_dir, _compute)

    def embed(self, samples: Sequence[EvaluationSample]) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for sample in samples:
                tile_out = self._tile_outputs_for_slide(sample.image_path)
                slide_out = pipeline.run_inference_with_slide_encoder(
                    slide_encoder_model=self.slide_encoder,
                    **tile_out,
                )
                vec = slide_out["last_layer_embed"]
                if isinstance(vec, torch.Tensor):
                    vec = vec.squeeze(0)
                else:
                    vec = torch.as_tensor(vec).squeeze(0)
                if self.normalize_output:
                    vec = F.normalize(vec, p=2, dim=-1)
                results[sample.sample_id] = vec.detach().cpu().numpy()
        return results


class CpGPTBaseEmbeddingExtractor:
    """
    Extract sample embeddings directly from CpGPT (no CLIP projection head).
    Reads the CpGPT dependencies root from the Lightning hparams.yaml used to train CLIP.
    """

    def __init__(self, config: ClipModelConfig):
        self.device = _default_device(config.device)
        self.normalize_output: bool = config.normalize_output
        self.batch_size: int = max(1, config.batch_size)

        arch = _load_arch_from_hparams(config.hparams_path)
        # path to cpgpt_files root used during training
        cpgpt_root = arch["dnameth_enc_path"]
        model_name = arch.get("dnameth_enc_name", "cancer")
        # Runner has its own caching for per-file embeddings
        self.runner = CpGPTRunner(root=cpgpt_root, model_name=str(model_name), device=str(self.device), precision="16-mixed", cache_embeddings=True)

    def embed(self, samples: Sequence[EvaluationSample]) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch_samples = samples[start : start + self.batch_size]
                paths = [s.dnameth_path for s in batch_samples]
                emb = self.runner.encode_beta_files(paths)  # [B, D]
                if self.normalize_output:
                    emb = F.normalize(emb, p=2, dim=-1)
                for sample, vector in zip(batch_samples, emb, strict=True):
                    results[sample.sample_id] = vector.detach().cpu().numpy()
        return results
