from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hescape.constants import DatasetEnum


# --------------------------- utilities ---------------------------

def set_worker_seed(worker_id: int):
    """Reproducible workers."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _resolve(path: str | Path, root: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else Path(root) / p)


def _read_betas_txt(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust reader for sesame level3betas files.
    Returns (probe_ids[str], betas[float]).
    """
    # Fast path with pandas
    try:
        df = pd.read_csv(txt_path, sep="\t", header=None, comment="#", dtype={0: str}, engine="c")
        # heuristic: if first row contains non numeric in col 1, treat first row as header
        if df.shape[1] >= 2 and not pd.api.types.is_numeric_dtype(df.iloc[:5, 1]):
            df = pd.read_csv(txt_path, sep="\t", header=0, comment="#", dtype={0: str}, engine="c")
        # keep first two columns
        df = df.iloc[:, :2]
        # coerce numeric for beta
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df.dropna(subset=[df.columns[1]], inplace=True)
        probes = df.iloc[:, 0].astype(str).values
        betas = df.iloc[:, 1].astype(np.float32).values
        return probes, betas
    except Exception:
        # Very robust fallback
        ids: List[str] = []
        vals: List[float] = []
        with open(txt_path, "r") as fh:
            for ln in fh:
                if not ln or ln[0] == "#":
                    continue
                parts = ln.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    val = float(parts[1])
                except Exception:
                    continue
                ids.append(parts[0])
                vals.append(val)
        return np.array(ids, dtype=str), np.array(vals, dtype=np.float32)


# --------------------------- schema helpers ---------------------------

@dataclass
class SampleRow:
    """Minimal fields this dataset needs from each JSONL line."""
    name: str
    img_tile_path: str
    beta_txt_path: str
    source: Optional[str] = None
    tissue: Optional[str] = None
    sample_type: Optional[str] = None  # for splitting if available

    @staticmethod
    def from_jsonl_obj(obj: Dict[str, Any], root: str) -> "SampleRow":
        # prefer sample_submitter_id then case_submitter_id
        name = obj.get("sample_submitter_id") or obj.get("case_submitter_id") or obj.get("case_id")
        # slide tile path
        slide = obj.get("slide", {})
        tile_rel = slide.get("tile_local_path") or slide.get("tile_path") or slide.get("local_tile_path")
        if tile_rel is None:
            raise KeyError("Missing slide.tile_local_path in JSONL row")
        img_tile_path = _resolve(tile_rel, root)

        # methylation beta txt
        meth = obj.get("methylation_beta", {})
        beta_rel = meth.get("local_path") or meth.get("file_path")
        if beta_rel is None:
            raise KeyError("Missing methylation_beta.local_path in JSONL row")
        beta_txt_path = _resolve(beta_rel, root)

        source = obj.get("project_id") or obj.get("primary_site")
        tissue = obj.get("primary_site") or obj.get("disease_type")
        sample_type = obj.get("sample_type") or obj.get("binary_label")

        return SampleRow(
            name=str(name),
            img_tile_path=img_tile_path,
            beta_txt_path=beta_txt_path,
            source=(source if source is None else str(source)),
            tissue=(tissue if tissue is None else str(tissue)),
            sample_type=(sample_type if sample_type is None else str(sample_type)),
        )


# --------------------------- dataset ---------------------------

class DnaMethWSIDataset(Dataset):
    """
    Dataset that yields a dict keyed by DatasetEnum:
      - NAME:   List[str] at collate, here str per item
      - IMG:    str path to tile JSON used by GigaPath (model side reads it)
      - GEXP:   torch.FloatTensor [N_sites] of beta values with NaN for missing
      - SOURCE: str if present
      - TISSUE: str if present

    Notes
    -----
    - Builds a global CpG site order at init by scanning all beta files.
    - Keeps an in-memory cache of per-sample beta vectors to speed up epochs.
    """

    def __init__(
        self,
        rows: List[SampleRow],
        min_site_presence: float = 0.0,  # keep sites present in at least this fraction of samples
        max_sites: Optional[int] = None,  # optional cap if you want to subsample
        verbose: bool = True,
    ):
        super().__init__()
        assert len(rows) > 0, "Empty dataset"
        self.rows = rows
        self.verbose = verbose

        # Build global site order
        self.site_index, self.site_list = self._build_site_index(rows, min_site_presence, max_sites)
        self.n_sites = len(self.site_list)
        if self.verbose:
            print(f"[DNAMeth] Total samples={len(rows)}  CpG sites={self.n_sites}")

        # simple in-memory cache: idx -> np.ndarray [N_sites]
        self._beta_cache: Dict[int, np.ndarray] = {}

    @staticmethod
    def _build_site_index(
        rows: List[SampleRow],
        min_site_presence: float,
        max_sites: Optional[int],
    ) -> Tuple[Dict[str, int], List[str]]:
        counts: Counter[str] = Counter()
        # one pass to count how many files contain each CpG id
        for r in rows:
            try:
                ids, _ = _read_betas_txt(r.beta_txt_path)
                counts.update(set(map(str, ids)))
            except Exception:
                # skip files that fail to read
                pass

        if not counts:
            raise RuntimeError("No CpG probes discovered while scanning beta files.")

        n = len(rows)
        # keep by fraction threshold
        keep = [k for k, c in counts.items() if (c / n) >= min_site_presence]
        # deterministic order
        keep.sort()
        if max_sites is not None:
            keep = keep[:max_sites]

        site_index = {probe: i for i, probe in enumerate(keep)}
        return site_index, keep

    def __len__(self) -> int:
        return len(self.rows)

    def _betas_to_vector(self, ids: np.ndarray, vals: np.ndarray) -> np.ndarray:
        vec = np.full(self.n_sites, np.nan, dtype=np.float32)
        # filter only probes we track
        # vectorized map via index lookup
        for pid, val in zip(ids, vals):
            idx = self.site_index.get(str(pid))
            if idx is not None:
                vec[idx] = np.float32(val)
        return vec

    def _load_vector(self, idx: int) -> np.ndarray:
        if idx in self._beta_cache:
            return self._beta_cache[idx]
        row = self.rows[idx]
        ids, vals = _read_betas_txt(row.beta_txt_path)
        vec = self._betas_to_vector(ids, vals)
        self._beta_cache[idx] = vec
        return vec

    def __getitem__(self, idx: int) -> Dict[Any, Any]:
        row = self.rows[idx]
        vec = self._load_vector(idx)  # np.ndarray [N_sites] with NaNs

        # assemble item
        item: Dict[Any, Any] = {
            DatasetEnum.NAME: row.name,
            DatasetEnum.IMG: row.img_tile_path,        # listified in collate
            DatasetEnum.GEXP: torch.from_numpy(vec),   # stacked in collate to [B, N_sites]
            DatasetEnum.SOURCE: row.source if row.source is not None else "",
            DatasetEnum.TISSUE: row.tissue if row.tissue is not None else "",
        }
        return item


# --------------------------- collate ---------------------------

def dnameth_wsi_collate(batch: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """
    Collate into tensors and lists while keeping DatasetEnum keys.
    """
    names = [b[DatasetEnum.NAME] for b in batch]
    imgs = [b[DatasetEnum.IMG] for b in batch]
    srcs = [b[DatasetEnum.SOURCE] for b in batch]
    tissues = [b[DatasetEnum.TISSUE] for b in batch]
    gexp = torch.stack([b[DatasetEnum.GEXP] for b in batch], dim=0)  # [B, N_sites]
    out: Dict[Any, Any] = {
        DatasetEnum.NAME: names,
        DatasetEnum.IMG: imgs,
        DatasetEnum.SOURCE: srcs,
        DatasetEnum.TISSUE: tissues,
        DatasetEnum.GEXP: gexp,
    }
    return out


# --------------------------- datamodule ---------------------------

class DnaMethWSIDataModule(LightningDataModule):
    """
    Reads JSONL with paired slide tile JSON and DNA methylation beta TXT.
    Produces batches compatible with HEScape training loops:
      - IMG: list of tile JSON paths (for GigaPath)
      - GEXP: float tensor [B, N_sites] of beta values with NaNs
      - NAME, SOURCE, TISSUE: metadata lists
    """

    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,                  # base directory that JSONL local paths are relative to
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        val_ratio: float = 0.05,        # 5 percent eval set
        test_ratio: float = 0.0,        # optional
        min_site_presence: float = 0.0, # keep sites present in at least this fraction of samples
        max_sites: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_site_presence = min_site_presence
        self.max_sites = max_sites
        self.verbose = verbose

        # populated in setup
        self.train_ds: Optional[DnaMethWSIDataset] = None
        self.val_ds: Optional[DnaMethWSIDataset] = None
        self.test_ds: Optional[DnaMethWSIDataset] = None

    def prepare_data(self) -> None:
        # nothing to download
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        rows_all = [SampleRow.from_jsonl_obj(obj, root=self.root_dir) for obj in read_jsonl(self.jsonl_path)]
        assert len(rows_all) > 0, f"No samples found in {self.jsonl_path}"

        # stratified split on sample_type if available, else random
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(rows_all))

        # build groups
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, obj in enumerate(read_jsonl(self.jsonl_path)):
            st = obj.get("sample_type") or obj.get("binary_label") or "UNK"
            groups[str(st)].append(i)

        val_idx: List[int] = []
        test_idx: List[int] = []
        train_idx: List[int] = []
        for g, idxs in groups.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n = len(idxs)
            n_val = int(round(self.val_ratio * n))
            n_test = int(round(self.test_ratio * n)) if self.test_ratio > 0 else 0
            val_idx.extend(idxs[:n_val].tolist())
            test_idx.extend(idxs[n_val:n_val + n_test].tolist())
            train_idx.extend(idxs[n_val + n_test:].tolist())

        # sanity: non empty train and some val
        if len(val_idx) == 0:
            # fallback: 5 percent random if group strat failed
            rng.shuffle(idx)
            n_val = max(1, int(round(self.val_ratio * len(idx))))
            val_idx = idx[:n_val].tolist()
            train_idx = idx[n_val:].tolist()
            test_idx = []

        # slice rows
        rows_train = [rows_all[i] for i in train_idx]
        rows_val = [rows_all[i] for i in val_idx]
        rows_test = [rows_all[i] for i in test_idx] if len(test_idx) > 0 else None

        # Build datasets. Important: build site index on train set, then apply to val/test for stability.
        train_ds = DnaMethWSIDataset(
            rows_train,
            min_site_presence=self.min_site_presence,
            max_sites=self.max_sites,
            verbose=self.verbose,
        )
        # copy the site index to val/test to keep same order
        def make_with_index(rows: List[SampleRow]) -> DnaMethWSIDataset:
            ds = DnaMethWSIDataset(rows, min_site_presence=0.0, max_sites=None, verbose=False)
            ds.site_index = dict(train_ds.site_index)
            ds.site_list = list(train_ds.site_list)
            ds.n_sites = train_ds.n_sites
            return ds

        val_ds = make_with_index(rows_val)
        test_ds = make_with_index(rows_test) if rows_test is not None else None

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        if self.verbose:
            print(
                f"[DNAMeth] Split: train={len(self.train_ds)}  val={len(self.val_ds)}"
                + (f"  test={len(self.test_ds)}" if self.test_ds is not None else "")
            )
            print(f"[DNAMeth] CpG sites used: {self.train_ds.n_sites}")

    # ----------------- dataloaders -----------------

    def _dl(self, ds: Dataset, shuffle: bool, pin: bool, persist: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin,
            persistent_workers=persist,
            collate_fn=dnameth_wsi_collate,
            worker_init_fn=set_worker_seed,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._dl(self.train_ds, shuffle=True, pin=self.pin_memory, persist=self.persistent_workers)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._dl(self.val_ds, shuffle=False, pin=self.pin_memory, persist=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            # use val as test if not provided
            assert self.val_ds is not None
            return self._dl(self.val_ds, shuffle=False, pin=self.pin_memory, persist=False)
        return self._dl(self.test_ds, shuffle=False, pin=self.pin_memory, persist=False)

    def predict_dataloader(self) -> DataLoader:
        # full set for prediction = train + val + test if present
        # simple choice: return val loader here, adapt as needed
        assert self.val_ds is not None
        return self._dl(self.val_ds, shuffle=False, pin=False, persist=False)

    @property
    def input_genes(self) -> int:
        """Return number of CpG sites for model initialization."""
        if self.train_ds is not None:
            return self.train_ds.n_sites
        return 0  # Will be set after setup
