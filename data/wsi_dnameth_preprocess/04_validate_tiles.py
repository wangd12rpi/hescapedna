#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Iterable, List

from PIL import Image
from gigapath.pipeline import tile_one_slide
from tqdm import tqdm


# Compute project_root like your other scripts
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [rec for rec in iter_jsonl(path)]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(path)


def guess_tile_base_dir(root: Path, rec: Dict[str, Any]) -> Path:
    """
    Return the directory under which PNG tiles live for this record.
    Preference:
      1) slide.tile_local_path if present and exists
      2) fallback to <sample_dir>/tiles/output/<slide_name>
      3) if that does not exist, try <sample_dir>/tiles/output/<slide_stem>
      4) if both missing, return the preferred path even if it does not exist
    """
    slide = rec["slide"]
    slide_rel = slide["local_path"]
    slide_path = root / slide_rel
    sample_dir = slide_path.parent
    tiles_root = sample_dir / "tiles"

    t_local = slide.get("tile_local_path")
    if t_local:
        p = root / t_local
        if p.exists():
            return p

    slide_name = Path(slide_rel).name
    p1 = tiles_root / "output" / slide_name
    if p1.exists():
        return p1

    p2 = tiles_root / "output" / Path(slide_name).stem
    if p2.exists():
        return p2

    return p1  # preferred canonical location even if missing


def list_pngs(tile_base_dir: Path) -> List[Path]:
    pngs =  sorted(tile_base_dir.rglob("*.png"))
    return pngs

def png_is_ok(p: Path) -> bool:
    """
    A PNG is valid if it can be opened and fully loaded and has nonzero dimensions.
    Minimal try/except to classify corruption without crashing the whole run.
    """
    try:
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            if w <= 0 or h <= 0:
                return False
        return True
    except Exception:
        return False


def slide_tiles_ok(tile_base_dir: Path) -> bool:
    pngs = list_pngs(tile_base_dir)
    if not pngs:
        return False
    for p in pngs:
        if not png_is_ok(p):
            return False
    return True


def remove_current_tiles(tile_base_dir: Path) -> None:
    # Remove only this slide's tile directory
    if tile_base_dir.exists():
        shutil.rmtree(tile_base_dir, ignore_errors=True)


def retile_once(root: Path, rec: Dict[str, Any]) -> None:
    """
    Retile a single slide once into <sample_dir>/tiles using the same call you used before.
    This is wrapped in a minimal try/except to make sure one failure does not stop the run.
    """
    slide_rel = rec["slide"]["local_path"]
    slide_path = root / slide_rel
    sample_dir = slide_path.parent
    tiles_root = sample_dir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)

    try:
        tile_one_slide(str(slide_path), save_dir=str(tiles_root), level=1)
    except Exception:
        # Treat a tiling exception as a failed attempt
        pass


def validate_or_retile(root: Path, rec: Dict[str, Any]) -> bool:
    """
    Returns True if this record ends up with valid tiles, False otherwise.
    """
    # First check
    base_dir = guess_tile_base_dir(root, rec)
    if slide_tiles_ok(base_dir):
        return True

    # Remove current tiles and retile once
    print("error in this", base_dir)
    remove_current_tiles(base_dir)
    retile_once(root, rec)

    # Recompute base_dir in case the tiler chose a different folder form
    base_dir = guess_tile_base_dir(root, rec)
    return slide_tiles_ok(base_dir)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PNG tiles from index_tile.jsonl. "
                    "If corrupt or missing, delete tiles, retile once, recheck, "
                    "then drop the record from index_tile.jsonl if still bad."
    )
    # Default to your train split under project_root
    parser.add_argument("--root", type=str, default=str(project_root / "data" / "test"),
                        help="Folder containing index_tile.jsonl and sample subfolders.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    index_tile = root / "index_tile.jsonl"
    if not index_tile.exists():
        print(f"Tile index not found: {index_tile}")
        return 1

    records = read_jsonl(index_tile)
    if not records:
        print("No records to validate.")
        return 0

    print(f"Validating {len(records)} tiled slides under {root} ...")
    kept: List[Dict[str, Any]] = []
    dropped = 0

    for rec in tqdm(records, desc="Validate and repair"):
        ok = validate_or_retile(root, rec)
        if ok:
            kept.append(rec)
        else:
            dropped += 1

    if dropped > 0:
        write_jsonl(index_tile, kept)
        print(f"Removed {dropped} record(s) from {index_tile} due to persistent tile corruption.")
    else:
        print("All records have valid PNG tiles.")

    print(f"Kept {len(kept)} of {len(records)} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
