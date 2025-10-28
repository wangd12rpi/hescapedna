from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from hescape.data_modules.dnameth_wsi_dataset import (
    _load_vocab_sites,
    _prepare_filtered_beta_file,
    _resolve,
    read_jsonl,
)


def _resolve_key_path(metadata: Mapping[str, Any], key_path: str, default: Any | None = None) -> Any | None:
    """Traverse nested dictionaries using a dotted key path."""
    value: Any = metadata
    for attr in key_path.split("."):
        if isinstance(value, Mapping) and attr in value:
            value = value[attr]
        else:
            return default
    return value


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """Container holding the assets associated with a single slide/methylation pair."""

    sample_id: str
    image_path: str
    dnameth_path: str
    metadata: Mapping[str, Any]

    def get(self, key_path: str, default: Any | None = None) -> Any | None:
        """Return a nested metadata field resolved via dotted path syntax."""
        return _resolve_key_path(self.metadata, key_path, default=default)


class SampleIndex:
    """Index of evaluation samples with helper methods for filtering and lookup."""

    def __init__(self, samples: Iterable[EvaluationSample]):
        self._samples: List[EvaluationSample] = list(samples)
        self._by_id: Dict[str, EvaluationSample] = {}
        for sample in self._samples:
            if sample.sample_id in self._by_id:
                raise ValueError(f"Duplicate sample_id detected: {sample.sample_id!r}")
            self._by_id[sample.sample_id] = sample

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def ids(self) -> List[str]:
        return list(self._by_id.keys())

    def get(self, sample_id: str) -> EvaluationSample:
        return self._by_id[sample_id]

    def select(self, sample_ids: Iterable[str]) -> List[EvaluationSample]:
        return [self._by_id[sid] for sid in sample_ids if sid in self._by_id]

    def filter(self, predicate: Callable[[EvaluationSample], bool]) -> "SampleIndex":
        return SampleIndex(sample for sample in self._samples if predicate(sample))

    def labels(self, key_path: str, default: Any | None = None) -> Dict[str, Any | None]:
        return {sample.sample_id: sample.get(key_path, default=default) for sample in self._samples}


def load_samples(
    index_path: str | Path,
    root_dir: str | Path,
    *,
    cpgpt_vocab_path: str | Path | None = None,
    processed_beta_dir: str | Path | None = None,
    dropna: bool = True,
) -> SampleIndex:
    """Load samples from a JSONL manifest into a convenient in-memory index."""

    index_path = Path(index_path)
    root_dir = Path(root_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"JSONL manifest not found: {index_path}")
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root folder not found: {root_dir}")

    rows = read_jsonl(index_path)
    vocab = _load_vocab_sites(cpgpt_vocab_path) if cpgpt_vocab_path else None
    processed_root = Path(processed_beta_dir) if processed_beta_dir else root_dir / "_processed_beta_eval"

    samples: List[EvaluationSample] = []
    for row in rows:
        sample_id = (
            row.get("sample_submitter_id")
            or row.get("case_submitter_id")
            or row.get("case_id")
            or row.get("slide", {}).get("file_id")
        )
        if sample_id is None:
            raise ValueError("Unable to derive a unique sample identifier from JSONL entry.")

        slide_info: MutableMapping[str, Any] = row.get("slide", {}) or {}
        tile_path = slide_info.get("tile_local_path") or slide_info.get("tile_path") or slide_info.get("local_path")
        if tile_path is None:
            raise ValueError(f"Missing tile path for sample {sample_id!r}")

        image_path = _resolve(tile_path, root_dir)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Tile directory for sample {sample_id!r} not found: {image_path}")

        meth_info: MutableMapping[str, Any] = row.get("methylation_beta", {}) or {}
        beta_path = meth_info.get("local_path") or meth_info.get("file_path")
        if beta_path is None:
            raise ValueError(f"Missing methylation beta file for sample {sample_id!r}")

        resolved_beta = _resolve(beta_path, root_dir)
        relative_beta = Path(beta_path)
        processed_beta = _prepare_filtered_beta_file(
            source_path=resolved_beta,
            relative_path=relative_beta,
            processed_root=processed_root,
            vocab=vocab,
            dropna=dropna,
        )
        samples.append(
            EvaluationSample(
                sample_id=str(sample_id),
                image_path=image_path,
                dnameth_path=processed_beta,
                metadata=row,
            )
        )

    return SampleIndex(samples)
