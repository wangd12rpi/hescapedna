from __future__ import annotations

import argparse
import json
import shutil
import multiprocessing
from functools import partial
from pathlib import Path
from typing import List, Dict, Any

from gigapath.pipeline import tile_one_slide
from tqdm import tqdm


def cleanup_previous_run(root: Path, index_tile_path: Path) -> None:
    """Deletes the tiled index and all tile directories."""
    print("--- Overwrite enabled: Cleaning up previous run ---")
    if index_tile_path.exists():
        print(f"Deleting {index_tile_path}")
        index_tile_path.unlink()

    for p in root.iterdir():
        if p.is_dir():
            tiles_dir = p / "tiles"
            if tiles_dir.exists() and tiles_dir.is_dir():
                print(f"Deleting {tiles_dir}")
                shutil.rmtree(tiles_dir)
    print("--- Cleanup complete ---")


def load_processed_ids(path: Path) -> set:
    """Reads the tile index file to find out which cases are already done."""
    if not path.exists():
        return set()

    with open(path, "r") as f:
        processed_ids = {json.loads(line)["case_id"] for line in f}

    print(f"Resuming. Found {len(processed_ids)} already tiled cases.")
    return processed_ids


def tile_slide(slide_path: Path, tiles_root: Path) -> Path:
    """Tiles a single slide and returns the save directory path."""
    slide_save_dir = tiles_root
    slide_save_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = slide_save_dir / "output" / slide_save_dir.name / "dataset.csv"
    if dataset_csv.exists():
        # This will print from multiple processes, so output may be interleaved
        # print(f"[SKIP] Already tiled: {slide_path}")
        return slide_save_dir

    # print(f"[TILE] {slide_path} -> {slide_save_dir} ")
    tile_one_slide(str(slide_path), save_dir=str(slide_save_dir), level=1)
    return slide_save_dir


def process_case(case_data: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """
    Worker function to process a single case.
    Takes a case dict and the root path, returns the updated dict.
    """
    slide_path = root / case_data["slide"]["local_path"]
    sample_dir = slide_path.parent
    tiles_root = sample_dir / "tiles"

    slide_save_dir = tile_slide(slide_path, tiles_root)
    slide_name = str(slide_path).split("/")[-1]
    tile_local_path = slide_save_dir.relative_to(root)
    case_data["slide"]["tile_local_path"] = str(
        tile_local_path) + "/output/" + slide_name

    return case_data


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tile slides based on an index file using multiple cores.")
    parser.add_argument("--root", type=str, default="../wsi_dnameth/train",
                        help="Root directory containing index.jsonl and sample subdirectories.")
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="Number of CPU cores to use for processing.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    index_jsonl_path = root / "index.jsonl"
    index_tile_jsonl_path = root / "index_tile.jsonl"

    overwrite = False
    if overwrite:
        cleanup_previous_run(root, index_tile_jsonl_path)

    processed_case_ids = load_processed_ids(index_tile_jsonl_path)

    with open(index_jsonl_path, "r") as f:
        all_cases = [json.loads(line) for line in f]

    # Filter out cases that have already been processed
    tasks = [case for case in all_cases if case["case_id"] not in processed_case_ids]

    if not tasks:
        print("No new cases to process. Everything is up to date. ✨")
        return 0

    print(f"Found {len(tasks)} new cases to tile.")

    # Use functools.partial to create a worker function with the 'root' arg pre-filled
    worker_func = partial(process_case, root=root)

    with open(index_tile_jsonl_path, "a") as outfile, multiprocessing.Pool(processes=args.workers) as pool:
        # Use pool.imap_unordered for efficiency and wrap with tqdm for a progress bar
        progress_bar = tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Tiling slides")

        for updated_case_data in progress_bar:
            outfile.write(json.dumps(updated_case_data) + "\n")
            outfile.flush()

    print("\nDone.")
    print(f"Processed {len(all_cases)} total cases from index.")
    print(f"Newly tiled {len(tasks)} cases in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
