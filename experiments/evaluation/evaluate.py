from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from hescape._utils import find_root
from hescape.evaluation import SampleIndex, load_samples


def _project_root() -> Path:
    return Path(find_root()).resolve()


def _load_cfg() -> dict:
    OmegaConf.register_new_resolver("project_root", lambda: str(_project_root()))
    cfg_path = _project_root() / "experiments" / "configuration" / "evaluate.yaml"
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_manifest(folder: Path) -> Dict[str, Path]:
    manifest = folder / "manifest.jsonl"
    mapping: Dict[str, Path] = {}
    if not manifest.exists():
        return mapping
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mapping[str(row["sample_id"])] = Path(row["path"])
    return mapping


def _load_vectors_for_samples(sample_ids: List[str], manifest: Mapping[str, Path]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for sid in sample_ids:
        p = manifest.get(sid)
        if p is None or not Path(p).exists():
            continue
        vec = torch.load(p, map_location="cpu")
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        else:
            vec = np.asarray(vec)
        result[sid] = vec
    return result


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_eval_nn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_state: int,
) -> Tuple[List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    aurocs: List[float] = []
    auprcs: List[float] = []

    for train_idx, test_idx in skf.split(x, y):
        x_train = torch.from_numpy(x[train_idx]).float().to(device)
        y_train = torch.from_numpy(y[train_idx]).float().to(device)
        x_test = torch.from_numpy(x[test_idx]).float().to(device)
        y_test = torch.from_numpy(y[test_idx]).float().to(device)

        model = SimpleMLP(in_dim=x.shape[1], hidden=256).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        ds = torch.utils.data.TensorDataset(x_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        model.train()
        for _ in range(epochs):
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
        aurocs.append(float(roc_auc_score(y_test.cpu().numpy(), probs)))
        auprcs.append(float(average_precision_score(y_test.cpu().numpy(), probs)))

    return aurocs, auprcs


def _collect_labels(dataset: SampleIndex, label_field: str, positive: str, negative: str) -> Tuple[List[str], np.ndarray]:
    labels = dataset.labels(label_field)
    sample_ids: List[str] = []
    targets: List[int] = []
    wanted = {positive, negative}
    for sid, label in labels.items():
        if label in wanted:
            sample_ids.append(sid)
            targets.append(1 if label == positive else 0)
    return sample_ids, np.array(targets, dtype=np.int64)


def main() -> None:
    cfg = _load_cfg()

    cache_dir = Path(cfg["cache"]["dir"]).resolve()
    out_dir = Path(cfg["output"]["dir"]).resolve()
    _ensure_dir(out_dir)

    # Which embedders and splits to evaluate
    embedders: List[str] = list(cfg.get("embedders", ["align", "gigapath", "cpgpt"]))
    splits: List[str] = list(cfg.get("splits", ["test"]))

    ds_cfg = cfg["dataset"]
    vocab = ds_cfg.get("cpgpt_vocab_path")
    proc_beta = ds_cfg.get("processed_beta_dir")
    dropna = bool(ds_cfg.get("dropna", True))

    results: List[dict] = []

    for split in splits:
        # Load dataset to obtain labels from JSONL
        if split not in ds_cfg:
            continue
        sc = ds_cfg[split]
        dataset = load_samples(
            index_path=sc["index_path"],
            root_dir=sc["root_dir"],
            cpgpt_vocab_path=vocab,
            processed_beta_dir=proc_beta,
            dropna=dropna,
        )

        # Preload manifests for all embedders in this split
        man_by_embedder: Dict[str, Dict[str, Path]] = {}
        for e in embedders:
            folder = cache_dir / split / e
            man_by_embedder[e] = _read_manifest(folder)

        # Tasks loop
        for task_name, task_cfg in cfg["tasks"].items():
            label_field = task_cfg["label_field"]
            positive = str(task_cfg["positive_label"])
            negative = str(task_cfg["negative_label"])
            folds = int(task_cfg.get("folds", 10))
            random_state = int(task_cfg.get("random_state", 42))

            # classifier params
            clf = cfg.get("classifier", {})
            epochs = int(clf.get("epochs", 50))
            batch_size = int(clf.get("batch_size", 64))
            lr = float(clf.get("lr", 3e-4))
            weight_decay = float(clf.get("weight_decay", 0.0))

            sample_ids, y = _collect_labels(dataset, label_field, positive, negative)
            if not sample_ids:
                continue

            for e in embedders:
                manifest = man_by_embedder.get(e, {})
                vecs_map = _load_vectors_for_samples(sample_ids, manifest)
                if not vecs_map:
                    continue
                # Align vectors to the sample_id order
                x = np.stack([vecs_map[sid] for sid in sample_ids if sid in vecs_map])
                y_eff = np.array([y[i] for i, sid in enumerate(sample_ids) if sid in vecs_map], dtype=np.int64)
                if x.shape[0] < 2 or x.ndim != 2:
                    continue

                aurocs, auprcs = _train_eval_nn(
                    x, y_eff,
                    folds=folds,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    random_state=random_state,
                )

                results.append({
                    "split": split,
                    "embedder": e,
                    "task": task_name,
                    "samples": int(x.shape[0]),
                    "summary": {
                        "auroc_mean": float(np.mean(aurocs)),
                        "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
                        "auprc_mean": float(np.mean(auprcs)),
                        "auprc_std": float(np.std(auprcs, ddof=1)) if len(auprcs) > 1 else 0.0,
                    },
                    "folds": [
                        {"fold": i + 1, "auroc": float(aurocs[i]), "auprc": float(auprcs[i])}
                        for i in range(len(aurocs))
                    ],
                })

    # Write results
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Evaluation results written to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()