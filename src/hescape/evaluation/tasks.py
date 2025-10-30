from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        clf: Dict[str, object] | None = None,
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
        # Classifier config
        self.clf = clf or {"name": "nn", "hidden_dim": 256, "dropout": 0.2, "epochs": 30, "batch_size": 64, "lr": 1e-3}

    def _train_eval_nn(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        hidden_dim: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> np.ndarray:
        in_dim = x_train.shape[1]

        class MLP(nn.Module):
            def __init__(self, d_in: int, d_hid: int, p: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_hid),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p),
                    nn.Linear(d_hid, 1),
                )

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.net(z).squeeze(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(in_dim, hidden_dim, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        x_train_t = torch.from_numpy(x_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        x_val_t = torch.from_numpy(x_val).float().to(device)

        # Simple mini-batch loop
        n = x_train_t.size(0)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_train_t.index_select(0, idx)
                yb = y_train_t.index_select(0, idx)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val_t)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs

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

        folds: List[FoldResult] = []
        clf_name = str(self.clf.get("name", "nn")).lower()

        for fold_index, (train_idx, test_idx) in enumerate(cv.split(matrix, y), start=1):
            x_tr, x_te = matrix[train_idx], matrix[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Standardize features fold-wise
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_te = scaler.transform(x_te)

            if clf_name == "logreg":
                clf = LogisticRegression(
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    solver="lbfgs",
                )
                clf.fit(x_tr, y_tr)
                probs = clf.predict_proba(x_te)[:, 1]
            else:
                # Simple NN
                probs = self._train_eval_nn(
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    hidden_dim=int(self.clf.get("hidden_dim", 256)),
                    dropout=float(self.clf.get("dropout", 0.2)),
                    epochs=int(self.clf.get("epochs", 30)),
                    batch_size=int(self.clf.get("batch_size", 64)),
                    lr=float(self.clf.get("lr", 1e-3)),
                )

            auroc = roc_auc_score(y_te, probs)
            auprc = average_precision_score(y_te, probs)
            folds.append(
                FoldResult(
                    fold=fold_index,
                    support=len(test_idx),
                    auroc=float(auroc),
                    auprc=float(auprc),
                    y_true=y_te.tolist(),
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
