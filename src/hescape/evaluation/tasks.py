from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class FoldResult:
    fold: int
    support: int
    auroc: float
    auprc: float
    y_true: List[int] = field(default_factory=list)
    y_score: List[float] = field(default_factory=list)


@dataclass(slots=True)
class CrossValidationResult:
    task_name: str
    embedder_name: str
    folds: List[FoldResult]

    @property
    def summary(self) -> Dict[str, float]:
        auroc_values = [fold.auroc for fold in self.folds]
        auprc_values = [fold.auprc for fold in self.folds]
        return {
            "auroc_mean": float(np.mean(auroc_values)),
            "auroc_std": float(np.std(auroc_values, ddof=1)) if len(auroc_values) > 1 else 0.0,
            "auprc_mean": float(np.mean(auprc_values)),
            "auprc_std": float(np.std(auprc_values, ddof=1)) if len(auprc_values) > 1 else 0.0,
        }


class BinaryClassificationTask:
    """Binary classification with stratified K-fold evaluation."""

    def __init__(
        self,
        name: str,
        label_field: str,
        positive_label: str,
        negative_label: str,
        *,
        folds: int = 10,
        random_state: int = 42,
        stratified: bool = True,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
    ):
        self.name = name
        self.label_field = label_field
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.folds = folds
        self.random_state = random_state
        self.stratified = stratified
        self.max_iter = max_iter
        self.class_weight = class_weight

    def run(
        self,
        embedder_name: str,
        embeddings: Mapping[str, np.ndarray],
        labels: Mapping[str, str | None],
    ) -> CrossValidationResult:
        samples, targets = self._prepare_samples(embeddings, labels)
        if not samples:
            raise ValueError(f"No samples matched labels {self.positive_label!r}/{self.negative_label!r}")

        matrix = np.stack([embeddings[sid] for sid in samples])
        y = np.array(targets, dtype=int)

        cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        folds: List[FoldResult] = []
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(matrix, y), start=1):
            model.fit(matrix[train_idx], y[train_idx])
            probs = model.predict_proba(matrix[test_idx])[:, 1]
            targets_fold = y[test_idx]

            auroc = roc_auc_score(targets_fold, probs)
            auprc = average_precision_score(targets_fold, probs)
            folds.append(
                FoldResult(
                    fold=fold_index,
                    support=len(test_idx),
                    auroc=float(auroc),
                    auprc=float(auprc),
                    y_true=targets_fold.tolist(),
                    y_score=probs.tolist(),
                )
            )

        return CrossValidationResult(task_name=self.name, embedder_name=embedder_name, folds=folds)

    def _prepare_samples(
        self,
        embeddings: Mapping[str, np.ndarray],
        labels: Mapping[str, str | None],
    ) -> tuple[List[str], List[int]]:
        sample_ids: List[str] = []
        targets: List[int] = []
        for sample_id, label in labels.items():
            if sample_id not in embeddings or label is None:
                continue
            if label == self.positive_label:
                sample_ids.append(sample_id)
                targets.append(1)
            elif label == self.negative_label:
                sample_ids.append(sample_id)
                targets.append(0)
        return sample_ids, targets
