from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hescape.constants import DatasetEnum


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path: str | Path, root: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else Path(root) / p)


def _load_vocab_sites(vocab_path: str | Path) -> List[str]:
    with open(vocab_path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [str(x) for x in data]
    for key in ("input", "sites", "var_names", "features"):
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]
    raise ValueError(f"Unrecognized CpGPT vocabulary format in {vocab_path}")


def _prepare_filtered_beta_file(
    source_path: str,
    relative_path: Path,
    processed_root: Path,
    vocab: Optional[List[str]],
    dropna: bool,
) -> str:
    if vocab is None:
        return source_path

    processed_root.mkdir(parents=True, exist_ok=True)
    destination = processed_root / relative_path
    destination = destination.with_suffix(destination.suffix + ".filtered.tsv")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return str(destination)

    df = pd.read_csv(
        source_path,
        sep="\t",
        header=None,
        names=["site", "beta"],
        na_values=["NA", "NaN", "nan", ""],
        dtype={"site": str, "beta": float},
    )
    cat = pd.CategoricalDtype(categories=vocab, ordered=True)
    df["site"] = df["site"].astype(cat)
    df = df.sort_values("site")
    df = df[~df["site"].isna()]
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    if dropna:
        df = df.dropna(subset=["beta"])
    # Persist in CpGPT-compatible two-column TSV
    df.to_csv(destination, sep="\t", header=False, index=False, float_format="%.6f")
    return str(destination)


class _SimpleDataset(Dataset):
    def __init__(self, samples: List[Dict[DatasetEnum, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[DatasetEnum, Any]:
        return self.samples[index]


def dnameth_wsi_collate(batch: List[Dict[DatasetEnum, Any]]) -> Dict[DatasetEnum, Any]:
    meth_items = [item[DatasetEnum.DNAMETH] for item in batch]
    if isinstance(meth_items[0], torch.Tensor):
        meth = torch.stack(meth_items, dim=0)
    else:
        meth = meth_items

    return {
        DatasetEnum.NAME: [item[DatasetEnum.NAME] for item in batch],
        DatasetEnum.IMG: [item[DatasetEnum.IMG] for item in batch],
        DatasetEnum.SOURCE: [item[DatasetEnum.SOURCE] for item in batch],
        DatasetEnum.TISSUE: [item[DatasetEnum.TISSUE] for item in batch],
        DatasetEnum.DNAMETH: meth,
    }


class DnaMethWSIDataModule(LightningDataModule):
    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        val_ratio: float = 0.05,
        test_ratio: float = 0.0,
        cpgpt_vocab_path: str | None = None,
        processed_beta_dir: str | None = None,
        dropna: bool = True,
        **unused_kwargs: Any,
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
        self.dropna = dropna
        self.cpgpt_vocab_path = cpgpt_vocab_path
        self.processed_beta_dir = processed_beta_dir

        self._vocab_sites: Optional[List[str]] = None
        self._processed_root: Optional[Path] = None

        self.train_ds: Optional[_SimpleDataset] = None
        self.val_ds: Optional[_SimpleDataset] = None
        self.test_ds: Optional[_SimpleDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.cpgpt_vocab_path:
            self._vocab_sites = _load_vocab_sites(self.cpgpt_vocab_path)
            processed_root = Path(self.processed_beta_dir) if self.processed_beta_dir else Path(
                self.root_dir) / "_processed_beta"
            self._processed_root = processed_root

        rows = read_jsonl(self.jsonl_path)
        samples: List[Dict[DatasetEnum, Any]] = []

        for obj in rows:
            slide = obj.get("slide", {})
            meth = obj.get("methylation_beta", {})
            name = obj.get("sample_submitter_id") or obj.get("case_submitter_id") or obj.get("case_id")

            tile_path = slide.get("tile_local_path") or slide.get("local_path")
            beta_path = meth.get("local_path") or meth.get("file_path")
            if tile_path is None or beta_path is None:
                raise ValueError(f"Missing tile or methylation path for sample {name!r}")

            relative_beta = Path(beta_path)
            sample = {
                DatasetEnum.NAME: str(name),
                DatasetEnum.IMG: _resolve(tile_path, self.root_dir),
                DatasetEnum.SOURCE: str(obj.get("project_id") or ""),
                DatasetEnum.TISSUE: str(obj.get("primary_site") or ""),
            }

            resolved_beta = _resolve(beta_path, self.root_dir)

            processed_root = self._processed_root or Path(self.root_dir) / "_processed_beta"
            processed_path = _prepare_filtered_beta_file(
                source_path=resolved_beta,
                relative_path=relative_beta,
                processed_root=processed_root,
                vocab=self._vocab_sites,
                dropna=self.dropna,
            )
            sample[DatasetEnum.DNAMETH] = processed_path

            samples.append(sample)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if samples:
            order = torch.randperm(len(samples), generator=generator).tolist()
            samples = [samples[i] for i in order]

        val_count = int(len(samples) * self.val_ratio)
        test_count = int(len(samples) * self.test_ratio)

        val_samples = samples[:val_count]
        test_samples = samples[val_count:val_count + test_count] if test_count else []
        train_samples = samples[val_count + test_count:]

        self.train_ds = _SimpleDataset(train_samples)
        self.val_ds = _SimpleDataset(val_samples) if val_samples else None
        self.test_ds = _SimpleDataset(test_samples) if test_samples else None

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=dnameth_wsi_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            return self._make_loader(self.train_ds, shuffle=False)
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        target = self.test_ds or self.val_ds or self.train_ds
        return self._make_loader(target, shuffle=False)
