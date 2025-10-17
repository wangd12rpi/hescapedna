#!/usr/bin/env python3
"""
Create a paired histology + DNA methylation JSONL index from GDC, then split into train and test.

Outputs:
- project_root/data/index_full.jsonl
- project_root/data/train/index.jsonl
- project_root/data/test/index.jsonl

local_path policy:
- slide.local_path and methylation_beta.local_path are relative to the split folder (train or test)
- We use the server metadata file_name to form deterministic destinations:
  * If file_name ends with ".gz" but not ".tar.gz": local_path = "<sample_id>/<file_name without .gz>"
  * Else: local_path = "<sample_id>/<file_name>"  (archives kept as is)

By default we split by case to avoid leakage. You can switch to sample splitting with --split_unit sample.
"""

import os
import json
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
import requests
from pathlib import Path

# -----------------------------
# Project root like original
# -----------------------------
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

# -----------------------------
# GDC endpoints
# -----------------------------
FILES = "https://api.gdc.cancer.gov/files"
CASES = "https://api.gdc.cancer.gov/cases"

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def tumor_normal_binary(sample_type: Optional[str]) -> str:
    if not sample_type:
        return "other"
    s = sample_type.lower()
    if "normal" in s or "control" in s:
        return "normal"
    if "tumor" in s or "metastatic" in s or "recurrent" in s:
        return "tumor"
    return "other"

def sort_files_first(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: ((x.get("file_name") or ""), (x.get("file_id") or "")))

def _request_with_retries(method: str, url: str, *, json_body=None, timeout: int = 120, max_retries: int = 5) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            if method == "POST":
                r = requests.post(url, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
    raise last_exc

def gdc_post_all(endpoint: str, payload: Dict[str, Any], page_size: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    offset = 0
    while True:
        body = dict(payload)
        body["size"] = page_size
        body["from"] = offset
        r = _request_with_retries("POST", endpoint, json_body=body)
        data = r.json().get("data", {})
        hits = data.get("hits", [])
        if not hits:
            break
        results.extend(hits)
        offset += page_size
    return results

def fetch_case_ids_with(file_filter: Dict[str, Any], *, program: str, project_id: Optional[str], page_size: int) -> Set[str]:
    content = [file_filter]
    if program:
        content.append({"op": "in", "content": {"field": "cases.project.program.name", "value": [program]}})
    if project_id:
        content.append({"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}})
    payload = {
        "filters": {"op": "and", "content": content},
        "fields": "cases.case_id"
    }
    case_ids: Set[str] = set()
    for f in gdc_post_all(FILES, payload, page_size):
        for c in f.get("cases", []) or []:
            cid = c.get("case_id")
            if cid:
                case_ids.add(cid)
    return case_ids

def files_for_case(case_id: str, extra_filters: List[Dict[str, Any]], page_size: int) -> List[Dict[str, Any]]:
    fields = ",".join([
        "file_id","file_name","md5sum","data_category","data_type","data_format",
        "experimental_strategy","workflow_type",
        "cases.submitter_id","cases.project.project_id",
        "cases.samples.submitter_id","cases.samples.sample_type"
    ])
    content = [{"op": "in", "content": {"field": "cases.case_id", "value": [case_id]}}] + extra_filters
    payload = {"filters": {"op": "and", "content": content}, "fields": fields}
    return gdc_post_all(FILES, payload, page_size)

def fetch_case_labels(case_id: str) -> Dict[str, Any]:
    fields = ",".join([
        "case_id","submitter_id","project.project_id","primary_site","disease_type",
        "diagnoses.primary_diagnosis","diagnoses.morphology",
        "diagnoses.tumor_stage","diagnoses.ajcc_pathologic_stage",
        "diagnoses.classification_of_tumor","diagnoses.diagnosis_is_primary_disease",
        "samples.submitter_id","samples.sample_type"
    ])
    payload = {"filters": {"op": "in", "content": {"field": "case_id", "value": [case_id]}},
               "fields": fields, "size": 1}
    r = _request_with_retries("POST", CASES, json_body=payload)
    hits = r.json().get("data", {}).get("hits", [])
    return hits[0] if hits else {}

def _attach(grouped, meta: Dict[str, Any], key: str) -> None:
    attached = False
    for c in meta.get("cases", []) or []:
        for s in c.get("samples", []) or []:
            sid = s.get("submitter_id")
            stype = s.get("sample_type")
            if sid:
                item = {
                    "file_id": meta["file_id"],
                    "file_name": meta.get("file_name"),
                    "md5sum": meta.get("md5sum"),
                    "data_type": meta.get("data_type"),
                    "data_category": meta.get("data_category"),
                    "data_format": meta.get("data_format"),
                    "experimental_strategy": meta.get("experimental_strategy"),
                    "workflow_type": meta.get("workflow_type")
                }
                if grouped[sid]["sample_type"] is None and stype:
                    grouped[sid]["sample_type"] = stype
                grouped[sid][key].append(item)
                attached = True
    if not attached and key == "slides":
        fn = meta.get("file_name", "")
        if fn.startswith("TCGA-"):
            sid = "-".join(fn.split(".")[0].split("-")[:4])
            item = {
                "file_id": meta["file_id"],
                "file_name": meta.get("file_name"),
                "md5sum": meta.get("md5sum"),
                "data_type": meta.get("data_type"),
                "data_category": meta.get("data_category"),
                "data_format": meta.get("data_format"),
                "experimental_strategy": meta.get("experimental_strategy"),
                "workflow_type": meta.get("workflow_type")
            }
            grouped[sid][key].append(item)

def _local_name_from_filename(file_name: str) -> str:
    # For .gz but not .tar.gz, predict post-decompression filename
    if file_name.endswith(".gz") and not file_name.endswith(".tar.gz"):
        return file_name[:-3]
    return file_name

def build_records_for_case(case_id: str, page_size: int) -> List[Dict[str, Any]]:
    slide_meta = files_for_case(case_id, [
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}}
    ], page_size)
    beta_meta = files_for_case(case_id, [
        {"op": "and", "content": [
            {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}}
        ]}
    ], page_size)

    grouped = defaultdict(lambda: {"sample_type": None, "slides": [], "betas": []})
    for m in slide_meta:
        _attach(grouped, m, "slides")
    for m in beta_meta:
        _attach(grouped, m, "betas")

    candidates = {sid: grp for sid, grp in grouped.items() if grp["slides"] and grp["betas"]}
    if not candidates:
        return []

    case_meta = fetch_case_labels(case_id)
    case_submitter = case_meta.get("submitter_id")
    project_id = (case_meta.get("project") or {}).get("project_id")
    primary_site = case_meta.get("primary_site")
    disease_type = case_meta.get("disease_type")
    diagnoses = case_meta.get("diagnoses") or []
    primary_dx = next((d for d in diagnoses if d.get("diagnosis_is_primary_disease")), diagnoses[0] if diagnoses else {})
    dx = {
        "primary_diagnosis": primary_dx.get("primary_diagnosis"),
        "morphology": primary_dx.get("morphology"),
        "ajcc_pathologic_stage": primary_dx.get("ajcc_pathologic_stage"),
        "tumor_stage": primary_dx.get("tumor_stage"),
        "classification_of_tumor": primary_dx.get("classification_of_tumor")
    }

    records: List[Dict[str, Any]] = []
    for sample_id, grp in candidates.items():
        slide = sort_files_first(grp["slides"])[0]
        beta  = sort_files_first(grp["betas"])[0]

        # Predict deterministic local names
        slide_name = slide.get("file_name") or slide["file_id"]
        beta_name  = beta.get("file_name") or beta["file_id"]
        slide_local = f"{sample_id}/{_local_name_from_filename(slide_name)}"
        beta_local  = f"{sample_id}/{_local_name_from_filename(beta_name)}"

        slide_out = dict(slide)
        slide_out["local_path"] = slide_local
        beta_out = dict(beta)
        beta_out["local_path"] = beta_local

        rec = {
            "case_id": case_id,
            "case_submitter_id": case_submitter,
            "project_id": project_id,
            "primary_site": primary_site,
            "disease_type": disease_type,
            "diagnosis": dx,
            "sample_submitter_id": sample_id,
            "sample_type": grp.get("sample_type"),
            "binary_label": tumor_normal_binary(grp.get("sample_type")),
            "slide": slide_out,
            "methylation_beta": beta_out
        }
        records.append(rec)

    return records

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build full paired index and split into train and test JSONLs.")
    parser.add_argument("--project_id", type=str, default="TCGA-BRCA", help="TCGA project id or None for all TCGA")
    parser.add_argument("--program", type=str, default="TCGA", help="Program filter")
    parser.add_argument("--max_cases", type=int, default=50, help="Optional limit on number of paired cases")
    parser.add_argument("--api_page_size", type=int, default=1000)
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction for split")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for splitting")
    parser.add_argument("--split_unit", type=str, choices=["case","sample"], default="case", help="Split unit")
    args = parser.parse_args()

    data_root = project_root / "data/wsi_dnameth"
    train_dir = data_root / "train"
    test_dir  = data_root / "test"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    # Find paired cases
    slide_cases = fetch_case_ids_with(
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        program=args.program, project_id=args.project_id, page_size=args.api_page_size
    )
    beta_cases = fetch_case_ids_with(
        {"op": "and", "content": [
            {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}}
        ]},
        program=args.program, project_id=args.project_id, page_size=args.api_page_size
    )
    paired_cases = sorted(slide_cases & beta_cases)
    if args.max_cases is not None:
        paired_cases = paired_cases[:args.max_cases]

    print(f"Found {len(paired_cases)} paired cases" + (f" in project {args.project_id}" if args.project_id else ""))

    # Build full record list
    full_records: List[Dict[str, Any]] = []
    for i, cid in enumerate(paired_cases, 1):
        print(f"[{i}/{len(paired_cases)}] case {cid}")
        recs = build_records_for_case(cid, page_size=args.api_page_size)
        full_records.extend(recs)

    # Write full JSONL
    full_path = data_root / "index_full.jsonl"
    with full_path.open("w") as fout:
        for rec in full_records:
            fout.write(json.dumps(rec) + "\n")
    print(f"Wrote {full_path.relative_to(project_root)}")

    # Split
    if args.split_unit == "case":
        # keep cases intact across splits
        case_to_recs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in full_records:
            case_to_recs[rec["case_id"]].append(rec)
        case_ids = list(case_to_recs.keys())
        rng = random.Random(args.seed)
        rng.shuffle(case_ids)
        n_train = max(1, int(round(len(case_ids) * args.train_frac))) if case_ids else 0
        train_cases = set(case_ids[:n_train])
        train_records = [r for r in full_records if r["case_id"] in train_cases]
        test_records  = [r for r in full_records if r["case_id"] not in train_cases]
    else:
        # sample-level shuffle
        rng = random.Random(args.seed)
        idxs = list(range(len(full_records)))
        rng.shuffle(idxs)
        n_train = max(1, int(round(len(full_records) * args.train_frac))) if full_records else 0
        train_idx = set(idxs[:n_train])
        train_records = [full_records[i] for i in range(len(full_records)) if i in train_idx]
        test_records  = [full_records[i] for i in range(len(full_records)) if i not in train_idx]

    # Write split JSONLs
    train_index = train_dir / "index.jsonl"
    test_index  = test_dir / "index.jsonl"

    with train_index.open("w") as ft:
        for rec in train_records:
            ft.write(json.dumps(rec) + "\n")
    with test_index.open("w") as fs:
        for rec in test_records:
            fs.write(json.dumps(rec) + "\n")

    print(f"Wrote {train_index.relative_to(project_root)} with {len(train_records)} samples")
    print(f"Wrote {test_index.relative_to(project_root)} with {len(test_records)} samples")

if __name__ == "__main__":
    main()
