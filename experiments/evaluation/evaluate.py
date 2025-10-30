from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from hescape._utils import find_root
from hescape.evaluation import (
    BinaryClassificationTask,
    CrossValidationResult,
    SampleIndex,
    load_samples,
)

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("project_root", lambda: find_root())


@dataclass(slots=True)
class JobSpec:
    name: str
    task: str
    embedder: str
    split: str
    sample_ids: List[str]
    labels: Mapping[str, str | None]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(str(path), allow_pickle=False)
    sample_ids: Sequence[str] = list(obj["sample_ids"])
    matrix = np.asarray(obj["embeddings"])
    return {sid: matrix[i] for i, sid in enumerate(sample_ids)}


def _save_results(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@hydra.main(config_path=".", config_name="evaluate", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Build task objects
    task_cfgs: Dict[str, DictConfig] = dict(cfg.tasks)
    tasks = {
        name: BinaryClassificationTask(
            name=name,
            label_field=task_cfgs[name].label_field,
            positive_label=str(task_cfgs[name].positive_label),
            negative_label=str(task_cfgs[name].negative_label),
            folds=int(task_cfgs[name].get("folds", 10)),
            random_state=int(task_cfgs[name].get("random_state", 42)),
            stratified=task_cfgs[name].get("stratified", True),
            max_iter=int(task_cfgs[name].get("max_iter", 1000)),
            clf=dict(cfg.evaluation.get("classifier", {"name": "nn"})),
        )
        for name in task_cfgs
    }

    # Load datasets per split for labels
    datasets: Dict[str, SampleIndex] = {}
    for split in cfg.dataset.splits:
        name = split.get("name")
        ds = load_samples(
            index_path=split.index_path,
            root_dir=split.root_dir,
            cpgpt_vocab_path=split.get("cpgpt_vocab_path"),
            processed_beta_dir=split.get("processed_beta_dir"),
            dropna=split.get("dropna", True),
        )
        datasets[name] = ds

    # Construct jobs
    jobs: List[JobSpec] = []
    for job_cfg in cfg.evaluation.jobs:
        task_name = job_cfg.task
        embedder = job_cfg.embedder
        split = job_cfg.get("split", cfg.dataset.get("default_split", "test"))
        job_name = job_cfg.name
        task = tasks[task_name]

        labels = datasets[split].labels(task.label_field)
        sample_ids = [sid for sid, val in labels.items() if val is not None]
        if not sample_ids:
            logger.warning("Skipping job '%s': no labels for task '%s' in split '%s'", job_name, task_name, split)
            continue

        jobs.append(JobSpec(name=job_name, task=task_name, embedder=embedder, split=split, sample_ids=sample_ids, labels=labels))

    if not jobs:
        logger.warning("No evaluation jobs scheduled; exiting.")
        return

    # Load embedding caches on demand
    cache_root = Path(cfg.cache.dir).resolve()
    results: List[dict] = []
    memo: Dict[tuple, Dict[str, np.ndarray]] = {}

    for job in jobs:
        cache_key = (job.split, job.embedder)
        if cache_key not in memo:
            npz_path = cache_root / job.split / job.embedder / "data.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing cache for split='{job.split}' embedder='{job.embedder}' at {npz_path}")
            memo[cache_key] = _load_npz(npz_path)

        embed_map = memo[cache_key]
        missing = [sid for sid in job.sample_ids if sid not in embed_map]
        if missing:
            logger.info("Job '%s': %d samples missing in cache, they will be ignored", job.name, len(missing))

        X = {sid: embed_map[sid] for sid in job.sample_ids if sid in embed_map}
        y = {sid: job.labels[sid] for sid in job.sample_ids if sid in embed_map}

        task = tasks[job.task]
        cv_result: CrossValidationResult = task.run(embedder_name=job.embedder, embeddings=X, labels=y)

        results.append(
            {
                "job": job.name,
                "task": job.task,
                "embedder": job.embedder,
                "split": job.split,
                "samples": len(X),
                "summary": cv_result.summary,
                "folds": [asdict(fold) for fold in cv_result.folds],
            }
        )

        summ = cv_result.summary
        logger.info(
            "[%s | %s | %s] AUROC=%.4f±%.4f AUPRC=%.4f±%.4f (n=%d)",
            job.name,
            job.task,
            job.embedder,
            summ["auroc_mean"],
            summ["auroc_std"],
            summ["auprc_mean"],
            summ["auprc_std"],
            len(X),
        )

    out_dir = Path(cfg.output.dir).resolve()
    _save_results(out_dir / "results.json", {"results": results})
    logger.info("Wrote evaluation results to %s", out_dir / "results.json")


if __name__ == "__main__":
    main()