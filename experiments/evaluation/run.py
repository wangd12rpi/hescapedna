from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hescape._utils import find_root
from hescape.evaluation import (
    BinaryClassificationTask,
    ClipFusionEmbeddingExtractor,
    ClipImageEmbeddingExtractor,
    ClipModelConfig,
    CrossValidationResult,
    SampleIndex,
    load_samples,
)

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("project_root", lambda: find_root())


@dataclass(slots=True)
class JobSpec:
    name: str
    embedder: str
    task: str
    sample_ids: List[str]
    labels: Mapping[str, str | None]


def _to_plain(obj: DictConfig | None) -> dict:
    if obj is None:
        return {}
    return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]


def _instantiate_embedder(
    name: str,
    embedder_cfg: DictConfig,
    clip_base_cfg: DictConfig,
) -> ClipFusionEmbeddingExtractor | ClipImageEmbeddingExtractor:
    embedder_type = embedder_cfg.get("type")
    base_model = _to_plain(clip_base_cfg.model)
    image_encoder = _to_plain(clip_base_cfg.image_encoder)
    dnameth_encoder = _to_plain(clip_base_cfg.dnameth_encoder)

    clip_config = ClipModelConfig(
        checkpoint_path=Path(clip_base_cfg.checkpoint_path),
        model=base_model,
        image_encoder=image_encoder,
        dnameth_encoder=dnameth_encoder,
        fusion=_to_plain(embedder_cfg.get("fusion")) if embedder_type == "clip_fusion" else None,
        device=embedder_cfg.get("device", clip_base_cfg.get("device")),
        batch_size=embedder_cfg.get("batch_size", clip_base_cfg.get("batch_size", 4)),
        normalize_output=embedder_cfg.get("normalize_output", True),
    )

    if embedder_type == "clip_fusion":
        return ClipFusionEmbeddingExtractor(clip_config)
    if embedder_type == "clip_image":
        return ClipImageEmbeddingExtractor(clip_config)
    raise ValueError(f"Unsupported embedder type {embedder_type!r} for embedder '{name}'")


def _instantiate_task(name: str, cfg: DictConfig) -> BinaryClassificationTask:
    task_type = cfg.get("type")
    if task_type != "binary_classification":
        raise ValueError(f"Unsupported task type {task_type!r} for task '{name}'")
    return BinaryClassificationTask(
        name=name,
        label_field=cfg.label_field,
        positive_label=str(cfg.positive_label),
        negative_label=str(cfg.negative_label),
        folds=int(cfg.get("folds", 10)),
        random_state=int(cfg.get("random_state", 42)),
        stratified=cfg.get("stratified", True),
        max_iter=int(cfg.get("max_iter", 1000)),
    )


def _collect_sample_ids(labels: Mapping[str, str | None], positive: str, negative: str) -> List[str]:
    wanted = {positive, negative}
    return [sample_id for sample_id, label in labels.items() if label in wanted]


def _serialize_result(job: JobSpec, outcome: CrossValidationResult) -> dict:
    return {
        "job": job.name,
        "embedder": job.embedder,
        "task": job.task,
        "samples": len(job.sample_ids),
        "summary": outcome.summary,
        "folds": [asdict(fold) for fold in outcome.folds],
    }


def _embedding_path(base_dir: Path, embedder_name: str) -> Path:
    return base_dir / f"{embedder_name}.pt"


def _save_embeddings(path: Path, mapping: Mapping[str, np.ndarray]) -> None:
    sample_ids = sorted(mapping.keys())
    if not sample_ids:
        raise ValueError("No embeddings to save.")
    matrix = np.stack([mapping[sid] for sid in sample_ids], axis=0).astype(np.float32)
    payload = {
        "sample_ids": sample_ids,
        "embeddings": torch.from_numpy(matrix),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sample_ids: List[str] = list(payload["sample_ids"])
    embeddings = payload["embeddings"]
    if isinstance(embeddings, torch.Tensor):
        matrix = embeddings.detach().cpu().numpy()
    else:
        matrix = np.asarray(embeddings)
    return {sid: matrix[idx] for idx, sid in enumerate(sample_ids)}


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(cfg.output_dir).resolve()
    embedding_dir = Path(cfg.embedding.dir).resolve()
    mode = cfg.get("mode", "both")

    dataset_cfg = cfg.dataset
    dataset: SampleIndex = load_samples(
        index_path=dataset_cfg.index_path,
        root_dir=dataset_cfg.root_dir,
        cpgpt_vocab_path=dataset_cfg.get("cpgpt_vocab_path"),
        processed_beta_dir=dataset_cfg.get("processed_beta_dir"),
        dropna=dataset_cfg.get("dropna", True),
    )
    logger.info("Loaded %d samples from %s", len(dataset), dataset_cfg.index_path)

    task_cfgs: Dict[str, DictConfig] = dict(cfg.tasks)
    tasks = {name: _instantiate_task(name, task_cfgs[name]) for name in task_cfgs}

    jobs: List[JobSpec] = []
    for job_cfg in cfg.evaluation.jobs:
        task_name = job_cfg.task
        embedder_name = job_cfg.embedder
        job_name = job_cfg.name
        task = tasks[task_name]

        labels = dataset.labels(task.label_field)
        sample_ids = _collect_sample_ids(labels, task.positive_label, task.negative_label)
        if not sample_ids:
            logger.warning("Skipping job '%s': no samples matched labels for task '%s'", job_name, task_name)
            continue

        job_spec = JobSpec(
            name=job_name,
            embedder=embedder_name,
            task=task_name,
            sample_ids=sample_ids,
            labels=labels,
        )
        jobs.append(job_spec)

    if not jobs:
        logger.warning("No evaluation jobs scheduled; nothing to do.")
        return

    required_embedder_names = {job.embedder for job in jobs}

    if mode in ("embed", "both"):
        embedder_cfgs: Dict[str, DictConfig] = dict(cfg.embedders)
        embedder_objects = {
            name: _instantiate_embedder(name, embedder_cfgs[name], cfg.clip_model) for name in required_embedder_names
        }
        sample_list = list(dataset)
        logger.info("Embedding stage: %d samples across %d embedders", len(sample_list), len(embedder_objects))
        for name, embedder in embedder_objects.items():
            target_path = _embedding_path(embedding_dir, name)
            if target_path.exists() and not cfg.embedding.get("force", False):
                logger.info("Skipping '%s' embeddings (found cached file at %s)", name, target_path)
                continue
            logger.info("Computing embeddings for '%s'", name)
            mapping = embedder.embed(sample_list)
            _save_embeddings(target_path, mapping)
        logger.info("Embedding stage complete. Files stored under %s", embedding_dir)
    else:
        embedding_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("eval", "both"):
        loaded_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        results: List[dict] = []
        for job in jobs:
            embedder_name = job.embedder
            if embedder_name not in loaded_embeddings:
                path = _embedding_path(embedding_dir, embedder_name)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Embeddings for embedder '{embedder_name}' not found at {path}. "
                        "Run with mode=embed or mode=both first."
                    )
                loaded_embeddings[embedder_name] = _load_embeddings(path)

            embedding_map = loaded_embeddings[embedder_name]
            missing = [sid for sid in job.sample_ids if sid not in embedding_map]
            if missing:
                logger.warning(
                    "%d samples required by job '%s' are missing from '%s' embeddings", len(missing), job.name, embedder_name
                )
            subset_embeddings = {sid: embedding_map[sid] for sid in job.sample_ids if sid in embedding_map}
            subset_labels = {sid: job.labels[sid] for sid in job.sample_ids}
            logger.info(
                "Running task '%s' using embedder '%s' on %d samples",
                job.task,
                job.embedder,
                len(subset_embeddings),
            )
            outcome = tasks[job.task].run(embedder_name=job.name, embeddings=subset_embeddings, labels=subset_labels)
            results.append(_serialize_result(job, outcome))

        if results:
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {"results": results}
            with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

            for item in results:
                summary = item["summary"]
                logger.info(
                    "[%s] AUROC=%.4f±%.4f, AUPRC=%.4f±%.4f over %d samples",
                    item["job"],
                    summary["auroc_mean"],
                    summary["auroc_std"],
                    summary["auprc_mean"],
                    summary["auprc_std"],
                    item["samples"],
                )


if __name__ == "__main__":
    main()
