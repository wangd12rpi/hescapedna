from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch


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


class _SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BinaryClassificationTask:
    """Binary classification with stratified K-fold evaluation.

    Supports either logistic regression ('logreg') or a simple MLP ('nn').
    """

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
        classifier: str = "logreg",
        # NN parameters (used when classifier='nn')
        nn_hidden_dim: int = 256,
        nn_epochs: int = 50,
        nn_batch_size: int = 64,
        nn_lr: float = 3e-4,
        nn_weight_decay: float = 0.0,
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
        self.classifier = classifier
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.nn_lr = nn_lr
        self.nn_weight_decay = nn_weight_decay

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

        if self.classifier == "nn":
            folds = self._run_nn(matrix, y)
        else:
            folds = self._run_logreg(matrix, y)

        return CrossValidationResult(task_name=self.name, embedder_name=embedder_name, folds=folds)

    def _run_logreg(self, matrix: np.ndarray, y: np.ndarray) -> List[FoldResult]:
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
        return folds

    def _run_nn(self, matrix: np.ndarray, y: np.ndarray) -> List[FoldResult]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)

        folds: List[FoldResult] = []
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(matrix, y), start=1):
            x_train = torch.from_numpy(matrix[train_idx]).float().to(device)
            y_train = torch.from_numpy(y[train_idx]).float().to(device)
            x_test = torch.from_numpy(matrix[test_idx]).float().to(device)
            y_test = torch.from_numpy(y[test_idx]).float().to(device)

            model = _SimpleMLP(in_dim=matrix.shape[1], hidden=self.nn_hidden_dim).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=self.nn_lr, weight_decay=self.nn_weight_decay)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            ds = torch.utils.data.TensorDataset(x_train, y_train)
            dl = torch.utils.data.DataLoader(ds, batch_size=self.nn_batch_size, shuffle=True, drop_last=False)

            model.train()
            for _ in range(self.nn_epochs):
                for xb, yb in dl:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(x_test).detach().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            auroc = roc_auc_score(y_test.cpu().numpy(), probs)
            auprc = average_precision_score(y_test.cpu().numpy(), probs)

            folds.append(
                FoldResult(
                    fold=fold_index,
                    support=len(test_idx),
                    auroc=float(auroc),
                    auprc=float(auprc),
                    y_true=y_test.cpu().numpy().astype(int).tolist(),
                    y_score=probs.tolist(),
                )
            )
        return folds

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
