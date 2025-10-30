from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Sequence, Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

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
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    clf_type: str,
    device: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    progress_desc: str,
) -> Any:
    """
    Train a classifier (logreg or simple NN) for binary or multiclass.
    Returns a fitted model. For NN, returns the torch model; for logreg, returns a sklearn pipeline.
    """
    if clf_type == "logreg":
        if num_classes == 2:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
                ]
            )
        else:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")),
                ]
            )
        model.fit(x, y)
        return model

    # NN
    dev = torch.device(device)
    out_dim = 1 if num_classes == 2 else num_classes
    model = SimpleMLP(in_dim=x.shape[1], out_dim=out_dim, hidden=hidden_dim).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if num_classes == 2:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float() if num_classes == 2 else torch.from_numpy(y).long()
    ds = torch.utils.data.TensorDataset(x_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in tqdm(range(epochs), desc=progress_desc):
        for xb, yb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits = model(xb).squeeze(-1) if num_classes == 2 else model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def _predict_proba(model: Any, x: np.ndarray, *, num_classes: int, device: str) -> np.ndarray:
    """
    Get class probabilities for binary or multiclass models.
    """
    if isinstance(model, Pipeline):
        proba = model.predict_proba(x)
        if num_classes == 2 and proba.ndim == 2 and proba.shape[1] == 2:
            return proba  # [N, 2]
        if num_classes == 2 and proba.ndim == 1:
            return np.stack([1 - proba, proba], axis=1)
        return proba

    # Torch model
    dev = torch.device(device)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().to(dev))
        if num_classes == 2:
            logits = logits.squeeze(-1)
            probs_pos = 1.0 / (1.0 + np.exp(-logits.detach().cpu().numpy()))
            probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)
        else:
            sm = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
            probs = sm
    return probs


# --------------------------------------------------------------------------------------
# Label prep
# --------------------------------------------------------------------------------------

def _prepare_binary_labels(index: SampleIndex, label_field: str, positive: str, negative: str) -> Tuple[List[str], np.ndarray]:
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
            f"No samples matched labels for label_field='{label_field}', positive='{positive}', negative='{negative}'."
        )
    return sample_ids, np.asarray(y, dtype=np.int64)


def _prepare_multiclass_labels(index: SampleIndex, label_field: str, classes: Sequence[str] | None = None) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Build a multiclass dataset from a single label field.
    If `classes` is None, infer from the data order encountered.
    Returns: sample_ids, y (int-coded), class_names (list index->name)
    """
    labels_map = index.labels(label_field)
    if classes is None:
        observed = [lbl for lbl in labels_map.values() if lbl is not None]
        classes = sorted(set(observed))
    class_to_id = {c: i for i, c in enumerate(classes)}
    sample_ids: List[str] = []
    y: List[int] = []
    for sid, label in labels_map.items():
        if label in class_to_id:
            sample_ids.append(sid)
            y.append(class_to_id[label])  # type: ignore[index]
    if not sample_ids:
        raise ValueError(f"No samples had labels among classes={classes!r} at field='{label_field}'.")
    return sample_ids, np.asarray(y, dtype=np.int64), list(classes)


# --------------------------------------------------------------------------------------
# Survival helpers
# --------------------------------------------------------------------------------------

def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _prepare_survival_arrays(index: SampleIndex, time_field: str, event_field: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Collect (sample_ids, time, event) arrays. time is float days (or given units).
    """
    times_map = index.labels(time_field)
    events_map = index.labels(event_field)
    sample_ids: List[str] = []
    times: List[float] = []
    events: List[int] = []
    for sid in index.ids():
        t = times_map.get(sid, None)
        e = events_map.get(sid, None)
        if t is None or e is None:
            continue
        try:
            tf = float(t)
        except Exception:
            continue
        ef = 1 if _parse_bool(e) else 0
        sample_ids.append(sid)
        times.append(tf)
        events.append(ef)
    if not sample_ids:
        raise ValueError(f"No samples with both time='{time_field}' and event='{event_field}'.")
    return sample_ids, np.asarray(times, dtype=float), np.asarray(events, dtype=np.int64)


def _harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Harrell's C-index for right-censored data.
    Pairs (i, j) are usable if the smaller time has an event=1.
    scores: higher means higher risk.
    O(N^2) implementation (sufficient for evaluation sizes).
    """
    n = len(times)
    assert times.shape == events.shape == (n,)
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j] and events[i] == events[j] == 0:
                continue
            if times[i] < times[j] and events[i] == 1:
                permissible += 1
                if scores[i] > scores[j]:
                    concordant += 1
                elif scores[i] == scores[j]:
                    ties += 1
            elif times[j] < times[i] and events[j] == 1:
                permissible += 1
                if scores[j] > scores[i]:
                    concordant += 1
                elif scores[j] == scores[i]:
                    ties += 1
    if permissible == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / permissible)


def _fit_survival_risk(
    x_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """
    Fit a Cox model if lifelines is available; otherwise use a simple
    logistic model on the event indicator as a fallback risk score proxy.
    Returns risk scores for x_test (higher = riskier).
    """
    try:
        import pandas as pd  # type: ignore
        from lifelines import CoxPHFitter  # type: ignore

        df_train = pd.DataFrame(x_train, columns=[f"x{i}" for i in range(x_train.shape[1])])
        df_train["time"] = t_train
        df_train["event"] = e_train
        cph = CoxPHFitter()
        cph.fit(df_train, duration_col="time", event_col="event", show_progress=False)

        df_test = pd.DataFrame(x_test, columns=[f"x{i}" for i in range(x_train.shape[1])])
        # lifelines partial hazards: higher = greater risk
        risk = np.asarray(cph.predict_partial_hazard(df_test)).reshape(-1)
        logging.info("Survival: used lifelines.CoxPHFitter for risk estimation.")
        return risk
    except Exception as ex:
        logging.warning("Survival fallback: lifelines not available or failed (%s). Using logistic risk proxy.", ex)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        )
        model.fit(x_train, e_train)
        # decision_function correlates with risk; if unavailable, use proba of event
        if hasattr(model["clf"], "decision_function"):
            # type: ignore[index]
            scores = model.decision_function(model["scaler"].transform(x_test))  # type: ignore[attr-defined]
            return np.asarray(scores).reshape(-1)
        proba = model.predict_proba(x_test)[:, 1]
        return proba.reshape(-1)


# --------------------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------------------

def _classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: Sequence[str],
    metric_list: Sequence[str],
) -> Dict[str, float]:
    """
    Compute a configurable set of classification metrics.
    proba: [N, C]
    """
    metrics: Dict[str, float] = {}
    num_classes = proba.shape[1]
    y_pred = np.argmax(proba, axis=1)

    # Helpers
    def safe(fn, default=np.nan):
        try:
            return float(fn())
        except Exception:
            return float(default)

    # AUCs / PRs require binarized labels for multiclass
    if num_classes > 2:
        Y = label_binarize(y_true, classes=list(range(num_classes)))
        auc_macro = safe(lambda: roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
        auc_weighted = safe(lambda: roc_auc_score(Y, proba, average="weighted", multi_class="ovr"))
        pr_macro = safe(lambda: average_precision_score(Y, proba, average="macro"))
        pr_weighted = safe(lambda: average_precision_score(Y, proba, average="weighted"))
    else:
        # binary: use positive class
        pos = proba[:, 1]
        auc_macro = safe(lambda: roc_auc_score(y_true, pos))
        auc_weighted = auc_macro
        pr_macro = safe(lambda: average_precision_score(y_true, pos))
        pr_weighted = pr_macro

    # Core metrics
    acc = safe(lambda: accuracy_score(y_true, y_pred))
    bacc = safe(lambda: balanced_accuracy_score(y_true, y_pred))
    f1_mac = safe(lambda: f1_score(y_true, y_pred, average="macro"))
    f1_mic = safe(lambda: f1_score(y_true, y_pred, average="micro"))
    f1_w = safe(lambda: f1_score(y_true, y_pred, average="weighted"))

    # Confusion matrix can be large; we export counts separately anyway
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics_blob = {
        "auroc_macro": auc_macro,
        "auroc_weighted": auc_weighted,
        "auprc_macro": pr_macro,
        "auprc_weighted": pr_weighted,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "f1_weighted": f1_w,
    }

    # only keep requested metrics
    for key in metric_list:
        if key in metrics_blob:
            metrics[key] = float(metrics_blob[key])

    # Always include a minimal confusion matrix summary for transparency
    metrics["_confusion_matrix"] = cm.tolist()
    return metrics


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

    # Global default metrics (optionally overridden per task)
    defaults = cfg.get("metrics", {})
    default_cls_metrics: List[str] = list(defaults.get("classification", ["auroc_macro", "auprc_macro", "accuracy", "f1_macro"]))
    default_surv_metrics: List[str] = list(defaults.get("survival", ["c_index"]))

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
        # Determine task type
        ttype: str = str(tcfg.get("type", "")).lower()
        # Backward-compat: if positive/negative present, it's binary classification
        if not ttype:
            if "positive_label" in tcfg and "negative_label" in tcfg:
                ttype = "binary"
            else:
                ttype = "multiclass"  # if user declares no pos/neg but a single label_field

        logging.info("Task '%s' -> type='%s'", task_name, ttype)

        # Shared label fields
        if ttype in {"binary", "binary_classification"}:
            label_field = tcfg["label_field"]
            pos = str(tcfg["positive_label"])
            neg = str(tcfg["negative_label"])

            train_ids, y_train = _prepare_binary_labels(train_index, label_field, pos, neg)
            test_ids, y_test = _prepare_binary_labels(test_index, label_field, pos, neg)

            # Counts by label for transparency
            def _counts_binary(y: np.ndarray) -> Dict[str, int]:
                c = Counter(y.tolist())
                return {"negative": int(c.get(0, 0)), "positive": int(c.get(1, 0))}

            label_counts_train = _counts_binary(y_train)
            label_counts_test = _counts_binary(y_test)
            class_names = [neg, pos]
            num_classes = 2

            for e in embedders:
                logging.info("Embedder '%s' :: loading vectors (train/test)", e)
                train_vecs_map = _load_vectors_for_samples(train_ids, manifests[("train", e)], desc=f"load {e} (train)")
                test_vecs_map = _load_vectors_for_samples(test_ids, manifests[("test", e)], desc=f"load {e} (test)")

                # Align order strictly to the sample_id lists (bugfix)
                x_train = np.stack([train_vecs_map[sid] for sid in train_ids])
                x_test = np.stack([test_vecs_map[sid] for sid in test_ids])

                model = _train_classifier(
                    x_train, y_train,
                    num_classes=num_classes,
                    clf_type=clf_type,
                    device=device,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    progress_desc=f"train [{task_name}|{e}]",
                )
                proba = _predict_proba(model, x_test, num_classes=num_classes, device=device)

                # Metrics
                metric_list = list(tcfg.get("metrics", default_cls_metrics))
                metrics = _classification_metrics(y_test, proba, class_names, metric_list)

                # Logging to console
                logging.info("RESULT [%s | %s] n_train=%d n_test=%d | counts_train=%s counts_test=%s",
                             task_name, e, x_train.shape[0], x_test.shape[0], label_counts_train, label_counts_test)
                for k, v in metrics.items():
                    if not k.startswith("_"):
                        logging.info("  %s: %.4f", k, v)

                item = {
                    "task": task_name,
                    "type": "binary",
                    "embedder": e,
                    "protocol": "train->test",
                    "samples": {"train": int(x_train.shape[0]), "test": int(x_test.shape[0])},
                    "classes": class_names,
                    "label_counts": {"train": label_counts_train, "test": label_counts_test},
                    "metrics": metrics,
                }
                results.append(item)

        elif ttype in {"multiclass", "multiclass_classification", "classification"}:
            label_field = tcfg["label_field"]
            classes = tcfg.get("classes", None)
            train_ids, y_train, class_names = _prepare_multiclass_labels(train_index, label_field, classes)
            test_ids, y_test, _ = _prepare_multiclass_labels(test_index, label_field, class_names)

            # Counts by label (name->count)
            def _counts(y: np.ndarray) -> Dict[str, int]:
                c = Counter(y.tolist())
                return {class_names[i]: int(c.get(i, 0)) for i in range(len(class_names))}

            label_counts_train = _counts(y_train)
            label_counts_test = _counts(y_test)

            num_classes = len(class_names)

            for e in embedders:
                logging.info("Embedder '%s' :: loading vectors (train/test)", e)
                train_vecs_map = _load_vectors_for_samples(train_ids, manifests[("train", e)], desc=f"load {e} (train)")
                test_vecs_map = _load_vectors_for_samples(test_ids, manifests[("test", e)], desc=f"load {e} (test)")

                x_train = np.stack([train_vecs_map[sid] for sid in train_ids])
                x_test = np.stack([test_vecs_map[sid] for sid in test_ids])

                model = _train_classifier(
                    x_train, y_train,
                    num_classes=num_classes,
                    clf_type=clf_type,
                    device=device,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    progress_desc=f"train [{task_name}|{e}]",
                )
                proba = _predict_proba(model, x_test, num_classes=num_classes, device=device)

                metric_list = list(tcfg.get("metrics", default_cls_metrics))
                metrics = _classification_metrics(y_test, proba, class_names, metric_list)

                logging.info("RESULT [%s | %s] n_train=%d n_test=%d | counts_train=%s counts_test=%s",
                             task_name, e, x_train.shape[0], x_test.shape[0], label_counts_train, label_counts_test)
                for k, v in metrics.items():
                    if not k.startswith("_"):
                        logging.info("  %s: %.4f", k, v)

                item = {
                    "task": task_name,
                    "type": "multiclass",
                    "embedder": e,
                    "protocol": "train->test",
                    "samples": {"train": int(x_train.shape[0]), "test": int(x_test.shape[0])},
                    "classes": class_names,
                    "label_counts": {"train": label_counts_train, "test": label_counts_test},
                    "metrics": metrics,
                }
                results.append(item)

        elif ttype in {"survival"}:
            time_field = tcfg["time_field"]
            event_field = tcfg["event_field"]
            metric_list = list(tcfg.get("metrics", default_surv_metrics))

            train_ids_all, t_train_all, e_train_all = _prepare_survival_arrays(train_index, time_field, event_field)
            test_ids_all, t_test_all, e_test_all = _prepare_survival_arrays(test_index, time_field, event_field)

            # Keep shared code path for embedders
            for e in embedders:
                logging.info("Embedder '%s' :: loading vectors (train/test)", e)
                train_vecs_map = _load_vectors_for_samples(train_ids_all, manifests[("train", e)], desc=f"load {e} (train)")
                test_vecs_map = _load_vectors_for_samples(test_ids_all, manifests[("test", e)], desc=f"load {e} (test)")

                # Align order
                x_train = np.stack([train_vecs_map[sid] for sid in train_ids_all])
                x_test = np.stack([test_vecs_map[sid] for sid in test_ids_all])

                # Train model -> risk scores
                risk_scores = _fit_survival_risk(x_train, t_train_all, e_train_all, x_test)

                # Metrics
                metrics: Dict[str, float] = {}
                if "c_index" in metric_list:
                    cidx = _harrell_c_index(t_test_all, e_test_all, risk_scores)
                    metrics["c_index"] = float(cidx)

                # Counts
                def _event_counts(events: np.ndarray) -> Dict[str, int]:
                    c = Counter(events.tolist())
                    return {"events": int(c.get(1, 0)), "censored": int(c.get(0, 0)), "n": int(events.shape[0])}

                counts_train = _event_counts(e_train_all)
                counts_test = _event_counts(e_test_all)

                logging.info("RESULT [%s | %s] n_train=%d n_test=%d | events_train=%s events_test=%s",
                             task_name, e, x_train.shape[0], x_test.shape[0], counts_train, counts_test)
                for k, v in metrics.items():
                    logging.info("  %s: %.4f", k, v)

                item = {
                    "task": task_name,
                    "type": "survival",
                    "embedder": e,
                    "protocol": "train->test",
                    "samples": {"train": int(x_train.shape[0]), "test": int(x_test.shape[0])},
                    "events": {"train": counts_train, "test": counts_test},
                    "metrics": metrics,
                }
                results.append(item)

        else:
            raise ValueError(f"Unsupported task type: {ttype!r}")

    # Write results (no silent fallbacks)
    _ensure_dir(out_dir)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    logging.info("Evaluation results written to %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
