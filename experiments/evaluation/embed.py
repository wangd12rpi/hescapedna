from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from hescape._utils import find_root
from hescape.evaluation import (
    ClipFusionEmbeddingExtractor,
    ClipImageEmbeddingExtractor,
    ClipDnaEmbeddingExtractor,
    ClipModelConfig,
    SampleIndex,
    load_samples,
)

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("project_root", lambda: find_root())


def _to_plain(obj: DictConfig | None) -> dict:
    if obj is None:
        return {}
    return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]


def _instantiate_clip_config(base_cfg: DictConfig, extra: Mapping[str, object] | None = None) -> ClipModelConfig:
    extras = {} if extra is None else dict(extra)
    base_model = _to_plain(base_cfg.model)
    image_encoder = _to_plain(base_cfg.image_encoder)
    dnameth_encoder = _to_plain(base_cfg.dnameth_encoder)

    return ClipModelConfig(
        checkpoint_path=Path(base_cfg.checkpoint_path),
        model=base_model,
        image_encoder=image_encoder,
        dnameth_encoder=dnameth_encoder,
        fusion=extras.get("fusion"),  # only used by fusion extractor
        device=extras.get("device") or base_cfg.get("device"),
        batch_size=int(extras.get("batch_size") or base_cfg.get("batch_size", 4)),
        normalize_output=bool(extras.get("normalize_output", True)),
    )


def _save_embeddings(path: Path, mapping: Mapping[str, np.ndarray]) -> None:
    sample_ids = sorted(mapping.keys())
    matrix = np.stack([mapping[sid] for sid in sample_ids], axis=0).astype(np.float32)
    payload = {"sample_ids": sample_ids, "embeddings": matrix}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)  # .npz


def _embed_split(
    split_name: str,
    dataset: SampleIndex,
    cfg: DictConfig,
    *,
    out_root: Path,
    embedders: Dict[str, object],
    force: bool,
) -> None:
    logger.info("Embedding split '%s' with %d samples", split_name, len(dataset))
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for tag, extractor in embedders.items():
        target = split_dir / tag / "data.npz"
        if target.exists() and not force:
            logger.info("Skip %-8s for %-5s (cached at %s)", tag, split_name, target)
            continue

        logger.info("Compute %-8s for %-5s -> %s", tag, split_name, target)
        mapping = extractor.embed(list(dataset))
        _save_embeddings(target, mapping)
        logger.info("Saved %d embeddings to %s", len(mapping), target)


@hydra.main(config_path=".", config_name="embed", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Materialize embedders
    clip_cfg = cfg.clip_model
    embedders: Dict[str, object] = {}

    if cfg.embedders.get("align", {}).get("enable", True):
        fusion_cfg = dict(cfg.embedders.align)
        fusion_cfg["fusion"] = _to_plain(cfg.embedders.align.get("fusion"))
        fusion = ClipFusionEmbeddingExtractor(_instantiate_clip_config(clip_cfg, fusion_cfg))
        embedders["align"] = fusion

    if cfg.embedders.get("gigapath", {}).get("enable", True):
        gigapath = ClipImageEmbeddingExtractor(_instantiate_clip_config(clip_cfg, cfg.embedders.get("gigapath")))
        embedders["gigapath"] = gigapath

    if cfg.embedders.get("cpgpt", {}).get("enable", True):
        dna = ClipDnaEmbeddingExtractor(_instantiate_clip_config(clip_cfg, cfg.embedders.get("cpgpt")))
        embedders["cpgpt"] = dna

    # Iterate splits
    cache_root = Path(cfg.cache.dir).resolve()
    force = bool(cfg.cache.get("force", False))

    for split in cfg.dataset.splits:
        name = split.get("name")
        ds: SampleIndex = load_samples(
            index_path=split.index_path,
            root_dir=split.root_dir,
            cpgpt_vocab_path=split.get("cpgpt_vocab_path"),
            processed_beta_dir=split.get("processed_beta_dir"),
            dropna=split.get("dropna", True),
        )
        _embed_split(name, ds, cfg, out_root=cache_root, embedders=embedders, force=force)

    logger.info("Embedding stage complete. Cache at %s", cache_root)


if __name__ == "__main__":
    main()