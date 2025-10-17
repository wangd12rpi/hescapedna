#!/usr/bin/env python3
"""
Download files listed in data/train/index.jsonl and data/test/index.jsonl.

Safe to rerun. Skips files that already exist at the exact local_path.
Supports optional MD5 verification for integrity.

Behavior:
- Uses slide.file_id and methylation_beta.file_id to download from GDC /data endpoint
- Saves to project_root/data/{train|test}/<local_path> as defined in the JSONL
- If source file_name ends with ".gz" but not ".tar.gz", we auto-decompress into local_path
  and remove the .gz temp file
- If source is ".tar.gz" or any other type, we save as is to local_path

Recommended: keep --verify_md5 off by default to avoid hashing large files repeatedly.
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable
import requests
import tempfile
import shutil
import gzip

# -----------------------------
# Project root like original
# -----------------------------
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _download_with_retries(url: str, dst_tmp: Path, *, timeout: int, max_retries: int = 5, chunk: int = 1 << 20) -> None:
    last_exc = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with dst_tmp.open("wb") as out:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            out.write(part)
            return
        except Exception as e:
            last_exc = e
    raise last_exc

def _md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _needs_download(dst: Path) -> bool:
    return not dst.exists() or dst.stat().st_size == 0

def _should_gunzip(src_gz_name: str) -> bool:
    return src_gz_name.endswith(".gz") and not src_gz_name.endswith(".tar.gz")

def _download_one(file_id: str, src_file_name: str, dst_final: Path, *, timeout: int, verify_md5: bool, md5sum: str = None) -> bool:
    """
    Download one GDC file by file_id into dst_final.
    Returns True if a download occurred, False if skipped.
    """
    ensure_dir(dst_final.parent)

    if not _needs_download(dst_final):
        if verify_md5 and md5sum:
            try:
                got = _md5sum(dst_final)
                if got != md5sum:
                    print(f"MD5 mismatch for {dst_final.name}. Redownloading.")
                else:
                    return False
            except Exception:
                pass
        else:
            return False  # already present

    url = f"{DATA_ENDPOINT}/{file_id}"

    # Temp path
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "download.tmp"

        # Download to temp
        _download_with_retries(url, tmp_path, timeout=timeout)

        # Decompress .gz if needed
        if _should_gunzip(src_file_name):
            # dst_final is already the target without .gz extension
            gz_tmp = Path(td) / "src.gz"
            shutil.move(str(tmp_path), str(gz_tmp))
            with gzip.open(gz_tmp, "rb") as fin, dst_final.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            # Save as is
            shutil.move(str(tmp_path), str(dst_final))

    if verify_md5 and md5sum:
        try:
            got = _md5sum(dst_final)
            if got != md5sum:
                print(f"Warning: MD5 mismatch for {dst_final}. Expected {md5sum} got {got}.")
        except Exception:
            print(f"Warning: could not compute MD5 for {dst_final}")

    return True

def _download_for_record(rec: Dict[str, Any], split_dir: Path, *, timeout: int, verify_md5: bool) -> None:
    # Slide
    slide = rec["slide"]
    s_id = slide["file_id"]
    s_name = slide.get("file_name") or s_id
    s_md5 = slide.get("md5sum")
    s_rel = slide["local_path"]
    s_dst = split_dir / s_rel

    # Beta
    beta = rec["methylation_beta"]
    b_id = beta["file_id"]
    b_name = beta.get("file_name") or b_id
    b_md5 = beta.get("md5sum")
    b_rel = beta["local_path"]
    b_dst = split_dir / b_rel

    # Download each if needed
    _download_one(s_id, s_name, s_dst, timeout=timeout, verify_md5=verify_md5, md5sum=s_md5)
    _download_one(b_id, b_name, b_dst, timeout=timeout, verify_md5=verify_md5, md5sum=b_md5)

def main():
    parser = argparse.ArgumentParser(description="Download files based on train and test JSONLs, skipping existing.")
    parser.add_argument("--which", type=str, choices=["all","train","test"], default="all", help="Which split to download")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout seconds")
    parser.add_argument("--verify_md5", action="store_true", help="Verify MD5 of existing and downloaded files")
    args = parser.parse_args()

    data_root = project_root / "data/wsi_dnameth"
    splits = []
    if args.which in ("all","train"):
        splits.append(("train", data_root / "train", data_root / "train" / "index.jsonl"))
    if args.which in ("all","test"):
        splits.append(("test", data_root / "test", data_root / "test" / "index.jsonl"))

    for split_name, split_dir, index_path in splits:
        if not index_path.exists():
            print(f"Index not found for {split_name}: {index_path}")
            continue
        print(f"Processing {split_name} from {index_path.relative_to(project_root)}")

        count = 0
        for rec in _iter_jsonl(index_path):
            count += 1
            try:
                _download_for_record(rec, split_dir, timeout=args.timeout, verify_md5=args.verify_md5)
                if count % 25 == 0:
                    print(f"  {count} samples processed...")
            except KeyboardInterrupt:
                print("Interrupted by user.")
                sys.exit(1)
            except Exception as e:
                # Do not abort entire run, continue to next sample
                sid = rec.get("sample_submitter_id", "unknown")
                print(f"Error on sample {sid}: {e}")

        print(f"Finished {split_name}. Samples processed: {count}")

if __name__ == "__main__":
    main()
