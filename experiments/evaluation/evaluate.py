from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from hescape._utils import find_root
from hescape.evaluation import SampleIndex, load_samples


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(find_root()).resolve()


def _load_cfg() -> dict:
    """
    Strictly load experiments/eval_configs/evaluate.yaml with ${project_root} resolver.
    No in-code defaults: if the YAML is missing something, let it raise.
    """
    OmegaConf.register_new_resolver("project_root", lambda: str(_project_root()))
    cfg_path = _project_root() / "experiments" / "eval_configs" / "evaluate.yaml"
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

def _read_manifest(folder: Path) -> Dict[str, Path]:
    """
    Read a manifest.jsonl mapping sample_id -> path (torch saved tensor).
    Raises if not present or empty.
    """
    manifest = folder / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    mapping: Dict[str, Path] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = str(row["sample_id"])

            p = Path(row["path"])
            mapping[sid] = p
    if not mapping:
        raise ValueError(f"Manifest at {manifest} is empty.")
    return mapping


def _load_vectors_for_samples(sample_ids: List[str], manifest: Mapping[str, Path], *, desc: str) -> Dict[str, np.ndarray]:
    """
    Load vectors for a given list of sample_ids using a manifest mapping.
    Raises if any requested ID is missing or file does not exist.
    """
    result: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    bad_files: List[str] = []

    for sid in tqdm(sample_ids, desc=desc):
        p = manifest.get(sid)
        if p is None:
            missing.append(sid)
            continue
        if not Path(p).exists():
            bad_files.append(sid)
            continue
        vec = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(vec, torch.Tensor):
            arr = vec.detach().cpu().numpy()
        else:
            arr = np.asarray(vec)
        result[sid] = arr

    if missing:
        raise KeyError(f"Missing {len(missing)} sample vectors (first 10): {missing[:10]}")
    if bad_files:
        raise FileNotFoundError(f"{len(bad_files)} manifest paths do not exist on disk (first 10): {bad_files[:10]}")

    return result


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _prepare_binary_labels(index: SampleIndex, label_field: str, positive: str, negative: str) -> Tuple[List[str], np.ndarray]:
    """
    Collect sample_ids with labels strictly equal to positive/negative.
    Raises if none selected.
    """
    labels = index.labels(label_field)
    sample_ids: List[str] = []
    y: List[int] = []
    for sid, label in labels.items():
        if label == positive:
            sample_ids.append(sid)
            y.append(1)
        elif label == negative:
            sample_ids.append(sid)
            y.append(0)
    if not sample_ids:
        raise ValueError(
            f"No samples matched labels for label_field='{label_field}', "
            f"positive='{positive}', negative='{negative}'."
        )
    return sample_ids, np.asarray(y, dtype=np.int64)


def _train_nn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    progress_desc: str,
) -> SimpleMLP:
    """
    Train a simple MLP on (x, y) with a visible tqdm progress bar.
    All hyperparameters are read from YAML by the caller.
    """
    dev = torch.device(device)
    model = SimpleMLP(in_dim=x.shape[1], hidden=hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in tqdm(range(epochs), desc=progress_desc):
        for xb, yb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def _train_logreg(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Deterministic logistic regression training (no internal defaults).
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    model.fit(x, y)
    return model


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = _load_cfg()

    # Required top-level sections (let KeyError bubble if absent)
    cache_dir = Path(cfg["cache"]["dir"]).resolve()
    out_dir = Path(cfg["output"]["dir"]).resolve()
    embedders: List[str] = list(cfg["embedders"])
    splits: List[str] = list(cfg["splits"])
    assert "train" in splits and "test" in splits, "splits must include both 'train' and 'test'"

    ds_cfg = cfg["dataset"]

    # Load datasets exactly as described by YAML
    logging.info("Loading datasets...")
    train_index: SampleIndex = load_samples(
        index_path=ds_cfg["train"]["index_path"],
        root_dir=ds_cfg["train"]["root_dir"],
        cpgpt_vocab_path=ds_cfg["cpgpt_vocab_path"],
        processed_beta_dir=ds_cfg["processed_beta_dir"],
        dropna=ds_cfg["dropna"],
    )
    test_index: SampleIndex = load_samples(
        index_path=ds_cfg["test"]["index_path"],
        root_dir=ds_cfg["test"]["root_dir"],
        cpgpt_vocab_path=ds_cfg["cpgpt_vocab_path"],
        processed_beta_dir=ds_cfg["processed_beta_dir"],
        dropna=ds_cfg["dropna"],
    )
    logging.info("Loaded %d train samples and %d test samples", len(train_index), len(test_index))

    # Strict classifier config
    clf_cfg = cfg["classifier"]
    clf_type: str = str(clf_cfg["type"])           # "nn" | "logreg"
    device: str = str(clf_cfg["device"])
    hidden_dim: int = int(clf_cfg["hidden_dim"])
    epochs: int = int(clf_cfg["epochs"])
    batch_size: int = int(clf_cfg["batch_size"])
    lr: float = float(clf_cfg["lr"])
    weight_decay: float = float(clf_cfg["weight_decay"])

    # Preload manifests per split+embedder; raise if anything is missing
    logging.info("Loading embedding manifests...")
    manifests: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for split in ("train", "test"):
        for e in embedders:
            folder = cache_dir / split / e
            m = _read_manifest(folder)
            manifests[(split, e)] = m
            logging.info("Manifest loaded: split=%s embedder=%s (n=%d)", split, e, len(m))

    results: List[dict] = []

    # Evaluate each task with each embedder
    for task_name, tcfg in cfg["tasks"].items():
        label_field = tcfg["label_field"]
        pos = str(tcfg["positive_label"])
        neg = str(tcfg["negative_label"])

        logging.info("Task '%s' -> field='%s' (pos=%s, neg=%s)", task_name, label_field, pos, neg)

        # Label selection (raises if no samples)
        train_ids, y_train = _prepare_binary_labels(train_index, label_field, pos, neg)
        test_ids, y_test = _prepare_binary_labels(test_index, label_field, pos, neg)
        logging.info("Task '%s': %d train samples, %d test samples", task_name, len(train_ids), len(test_ids))

        for e in embedders:
            logging.info("Embedder '%s' :: loading vectors (train/test)", e)
            train_vecs_map = _load_vectors_for_samples(train_ids, manifests[("train", e)], desc=f"load {e} (train)")
            test_vecs_map = _load_vectors_for_samples(test_ids, manifests[("test", e)], desc=f"load {e} (test)")

            # Align order strictly to the sample_id lists (raises if any missing above)
            x_train = np.stack([train_vecs_map[sid] for sid in train_vecs_map])
            x_test = np.stack([test_vecs_map[sid] for sid in test_vecs_map])

            # Sanity checks (let shape errors throw)
            logging.info(
                "Shapes: x_train=%s, x_test=%s | y_train=%s, y_test=%s",
                tuple(x_train.shape), tuple(x_test.shape), y_train.shape, y_test.shape
            )

            # ------------------------------------------------------------------
            # Train
            # ------------------------------------------------------------------
            if clf_type == "nn":
                model = _train_nn(
                    x_train, y_train,
                    device=device,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    progress_desc=f"train NN [{task_name}|{e}]",
                )
                model.eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(x_test).float().to(device)).detach().cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
            elif clf_type == "logreg":
                model = _train_logreg(x_train, y_train)
                probs = model.predict_proba(x_test)[:, 1]
            else:
                raise ValueError(f"Unsupported classifier type: {clf_type!r}")

            # ------------------------------------------------------------------
            # Test metrics (raise if metrics error)
            # ------------------------------------------------------------------
            auroc = float(roc_auc_score(y_test, probs))
            auprc = float(average_precision_score(y_test, probs))

            item = {
                "task": task_name,
                "embedder": e,
                "protocol": "train->test",
                "samples": {
                    "train": int(x_train.shape[0]),
                    "test": int(x_test.shape[0]),
                },
                "metrics": {
                    "auroc": auroc,
                    "auprc": auprc,
                },
            }
            results.append(item)
            logging.info("RESULT [%s | %s] AUROC=%.4f, AUPRC=%.4f", task_name, e, auroc, auprc)

    # Write results (no silent fallbacks)
    _ensure_dir(out_dir)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    logging.info("Evaluation results written to %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
