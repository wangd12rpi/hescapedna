from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from hescape._utils import find_root
from hescape.evaluation import (
    ClipFusionEmbeddingExtractor,
    ClipImageEmbeddingExtractor,
    ClipDnaEmbeddingExtractor,
    ClipModelConfig,
    SampleIndex,
    load_samples,
    GigaPathBaseEmbeddingExtractor,
    CpGPTBaseEmbeddingExtractor,
)


def _project_root() -> Path:
    return Path(find_root()).resolve()


def _load_cfg() -> dict:
    # Read experiments/configuration/embed.yaml with ${project_root} resolver
    OmegaConf.register_new_resolver("project_root", lambda: str(_project_root()))
    cfg_path = _project_root() / "experiments" / "eval_configs" / "embed.yaml"
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_per_sample(folder: Path, mapping: Mapping[str, np.ndarray]) -> None:
    _ensure_dir(folder)
    manifest: List[dict] = []
    for sid, vec in tqdm(mapping.items()):
        out = folder / f"{sid}.pt"
        torch.save(torch.as_tensor(vec, dtype=torch.float32), out)
        manifest.append({"sample_id": sid, "path": str(out)})
    with (folder / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")


def _load_index(index_path: str | Path, root_dir: str | Path, cpgpt_vocab_path: str | Path | None,
                processed_beta_dir: str | Path | None, dropna: bool) -> SampleIndex:
    return load_samples(
        index_path=index_path,
        root_dir=root_dir,
        cpgpt_vocab_path=cpgpt_vocab_path,
        processed_beta_dir=processed_beta_dir,
        dropna=dropna,
    )


def _embed_split(
    name: str,
    dataset: SampleIndex,
    cache_dir: Path,
    clip_cfg: ClipModelConfig,
    selected: set[str],
) -> None:
    """
    Write four optional caches based on `selected`:
      - clip_gigapath: CLIP image branch (after head)
      - clip_cpgpt:   CLIP DNA branch (after head)
      - gigapath_base: direct GigaPath slide embeddings (no CLIP head)
      - cpgpt_base:    direct CpGPT sample embeddings (no CLIP head)
    """
    target = cache_dir / name

    if "clip_gigapath" in selected:
        image = ClipImageEmbeddingExtractor(clip_cfg)
        mapping = image.embed(list(dataset))
        _save_per_sample(target / "clip_gigapath", mapping)

    if "clip_cpgpt" in selected:
        dna = ClipDnaEmbeddingExtractor(clip_cfg)
        mapping = dna.embed(list(dataset))
        _save_per_sample(target / "clip_cpgpt", mapping)

    if "gigapath_base" in selected:
        gp = GigaPathBaseEmbeddingExtractor(clip_cfg)
        mapping = gp.embed(list(dataset))
        _save_per_sample(target / "gigapath_base", mapping)

    if "cpgpt_base" in selected:
        cp = CpGPTBaseEmbeddingExtractor(clip_cfg)
        mapping = cp.embed(list(dataset))
        _save_per_sample(target / "cpgpt_base", mapping)


def main() -> None:
    cfg = _load_cfg()

    cache_dir = Path(cfg["cache"]["dir"]).resolve()
    _ensure_dir(cache_dir)

    # Dataset sections: train and/or test are optional; process whichever exists
    ds_cfg = cfg["dataset"]
    common_beta_proc = ds_cfg.get("processed_beta_dir")
    vocab = ds_cfg.get("cpgpt_vocab_path")
    dropna = bool(ds_cfg.get("dropna", True))

    splits: Dict[str, dict] = {}
    # make keys explicit if present
    if "train" in ds_cfg:
        splits["train"] = ds_cfg["train"]
    if "test" in ds_cfg:
        splits["test"] = ds_cfg["test"]

    # CLIP instantiation purely from checkpoint/hparams
    clip_section = cfg["clip"]
    clip_cfg = ClipModelConfig(
        checkpoint_path=Path(clip_section["checkpoint_path"]),
        hparams_path=Path(clip_section["hparams_path"]),
        device=clip_section.get("device"),
        batch_size=int(clip_section.get("batch_size", 8)),
        normalize_output=bool(clip_section.get("normalize_output", True)),
    )

    # Which embedders to run
    embedders = set(cfg.get("embedders", []))

    for split_name, sc in splits.items():
        root_dir = sc["root_dir"]
        index_path = sc["index_path"]
        dataset = _load_index(
            index_path=index_path,
            root_dir=root_dir,
            cpgpt_vocab_path=vocab,
            processed_beta_dir=common_beta_proc,
            dropna=dropna,
        )
        _embed_split(
            split_name,
            dataset,
            cache_dir,
            clip_cfg,
            selected=embedders,
        )

    print(f"Embedding cache written under: {cache_dir}")


if __name__ == "__main__":
    main()
