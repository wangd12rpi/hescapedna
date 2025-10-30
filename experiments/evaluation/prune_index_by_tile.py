#!/usr/bin/env python3
"""
Temporary fix:
Prune `index.jsonl` so it only contains samples that also exist in `index_tile.jsonl`
for both TRAIN and TEST splits. This prevents evaluation from requesting embeddings
that were never produced by the embed stage.

Hardcoded paths relative to the repo layout:
  data/wsi_dnameth/train/index.jsonl
  data/wsi_dnameth/train/index_tile.jsonl
  data/wsi_dnameth/test/index.jsonl
  data/wsi_dnameth/test/index_tile.jsonl

Run:
  python experiments/evaluation/prune_index_by_tile.py

This script will:
  - Read sample IDs from {split}/index_tile.jsonl
  - Filter {split}/index.jsonl to keep only those sample IDs
  - Create a timestamped .bak of the original index.jsonl before overwriting
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set, Dict, Any


# --------------------------------------------------------------------------------------
# Hardcoded paths (relative to this file's location inside the repo)
# --------------------------------------------------------------------------------------

# Repo root inferred from this file's path (no CLI / config involved)
REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = REPO_ROOT / "data" / "wsi_dnameth" / "train"
TEST_DIR = REPO_ROOT / "data" / "wsi_dnameth" / "test"

TRAIN_INDEX_JSONL = TRAIN_DIR / "index.jsonl"
TRAIN_TILE_INDEX_JSONL = TRAIN_DIR / "index_tile.jsonl"

TEST_INDEX_JSONL = TEST_DIR / "index.jsonl"
TEST_TILE_INDEX_JSONL = TEST_DIR / "index_tile.jsonl"


# --------------------------------------------------------------------------------------
# JSONL utilities
# --------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# Sample-id extraction (keep parity with evaluation/data.py fallback order)
# --------------------------------------------------------------------------------------

def sample_id_from_row(row: Dict[str, Any]) -> str | None:
    """
    Try to derive a unique sample_id using the same fallback order used by the pipeline:
      - sample_submitter_id
      - case_submitter_id
      - case_id
      - slide.file_id
    """
    for key in ("sample_submitter_id", "case_submitter_id", "case_id"):
        val = row.get(key)
        if val is not None:
            return str(val)

    slide = row.get("slide") or {}
    val = slide.get("file_id")
    if val is not None:
        return str(val)

    return None


def collect_ids(jsonl_path: Path) -> Set[str]:
    rows = read_jsonl(jsonl_path)
    ids: Set[str] = set()
    for r in rows:
        sid = sample_id_from_row(r)
        if sid is not None:
            ids.add(sid)
    return ids


def filter_rows_by_ids(rows: List[Dict[str, Any]], allowed_ids: Set[str]) -> List[Dict[str, Any]]:
    keep: List[Dict[str, Any]] = []
    for r in rows:
        sid = sample_id_from_row(r)
        if sid is not None and sid in allowed_ids:
            keep.append(r)
    return keep


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".{ts}.bak")
    shutil.copy2(path, backup)
    return backup


def prune_split(split_name: str, index_path: Path, tile_index_path: Path) -> None:
    print(f"\n=== {split_name.upper()} ===")
    print(f"Index JSONL:      {index_path}")
    print(f"Tile Index JSONL: {tile_index_path}")

    tile_ids = collect_ids(tile_index_path)
    print(f"Found {len(tile_ids)} sample IDs in tile index")

    rows = read_jsonl(index_path)
    print(f"Loaded {len(rows)} rows from index.jsonl")

    kept = filter_rows_by_ids(rows, tile_ids)
    dropped = len(rows) - len(kept)

    if dropped == 0:
        print("No pruning needed; files are already consistent.")
        return

    backup = backup_file(index_path)
    write_jsonl(index_path, kept)

    print(f"Pruned {dropped} rows; kept {len(kept)} rows.")
    print(f"Original backed up to: {backup}")


def main() -> None:
    # Validate paths exist early
    for p in [TRAIN_INDEX_JSONL, TRAIN_TILE_INDEX_JSONL, TEST_INDEX_JSONL, TEST_TILE_INDEX_JSONL]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    prune_split("train", TRAIN_INDEX_JSONL, TRAIN_TILE_INDEX_JSONL)
    prune_split("test", TEST_INDEX_JSONL, TEST_TILE_INDEX_JSONL)

    print("\nDone. Re-run the embedding stage, then evaluation.")


if __name__ == "__main__":
    main()