from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from hescape._utils import find_root
from hescape.evaluation import (
    ClipFusionEmbeddingExtractor,
    ClipImageEmbeddingExtractor,
    ClipModelConfig,
    SampleIndex,
    load_samples,
)
from hescape.evaluation.embedder import ClipDnaEmbeddingExtractor  # new DNA-only extractor


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
    for sid, vec in mapping.items():
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
    normalize_output: bool,
    batch_size: int,
    do_align: bool,
    do_img: bool,
    do_dna: bool,
) -> None:
    target = cache_dir / name
    if do_align:
        fusion = ClipFusionEmbeddingExtractor(clip_cfg)
        mapping = fusion.embed(list(dataset))
        _save_per_sample(target / "align", mapping)

    if do_img:
        image = ClipImageEmbeddingExtractor(clip_cfg)
        mapping = image.embed(list(dataset))
        _save_per_sample(target / "gigapath", mapping)

    if do_dna:
        dna = ClipDnaEmbeddingExtractor(clip_cfg)
        mapping = dna.embed(list(dataset))
        _save_per_sample(target / "cpgpt", mapping)


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
    splits["train"] = ds_cfg["train"]
    splits["test"] = ds_cfg["test"]

    # CLIP instantiation purely from checkpoint
    clip_section = cfg["clip"]
    clip_cfg = ClipModelConfig(
        checkpoint_path=Path(clip_section["checkpoint_path"]),
        hparams_path=Path(clip_section["hparams_path"]),
        device=clip_section.get("device"),
        batch_size=int(clip_section.get("batch_size", 8)),
        normalize_output=bool(clip_section.get("normalize_output", True)),
    )

    # Which embedders to run
    embedders = set(cfg.get("embedders", ["align", "gigapath", "cpgpt"]))
    do_align = "align" in embedders
    do_img = "gigapath" in embedders
    do_dna = "cpgpt" in embedders

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
            normalize_output=clip_cfg.normalize_output,
            batch_size=clip_cfg.batch_size,
            do_align=do_align,
            do_img=do_img,
            do_dna=do_dna,
        )

    print(f"Embedding cache written under: {cache_dir}")


if __name__ == "__main__":
    main()
