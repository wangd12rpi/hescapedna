#!/usr/bin/env python3
"""
Create a paired histology + DNA methylation JSONL index from GDC, then split into train and test.
This version augments each sample with a structured `labels` block for downstream tasks while
preserving all legacy fields and local paths.

Outputs:
- project_root/data/wsi_dnameth/index_full.jsonl
- project_root/data/wsi_dnameth/train/index.jsonl
- project_root/data/wsi_dnameth/test/index.jsonl

Legacy compatibility:
- slide.local_path and methylation_beta.local_path are unchanged
- existing keys remain; only a new `labels` key is added per record

Downstream labels added:
- labels.wsi_slide: morphology, grade, stage, slide QC percents (when available)
- labels.brca_receptors: ER/PR/HER2 IHC status plus triple_negative flag (BRCA only, from BCR Biotab)
- labels.mutations: per-gene boolean flags from /analysis/ssm_occurrences
- labels.survival: vital_status, event, os_days, days_to_last_follow_up, days_to_death
- labels.exposure: cigarettes_per_day, years_smoked, pack_years_smoked, computed_pack_years

Debugging:
- pass --debug to print payloads, counts, and first examples for receptor and mutation queries
- mutation path prints a clear warning and filter payload if everything is False for a cohort
"""

import os
import json
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Iterable, Tuple
import requests
from pathlib import Path
import csv
import io
import math

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
DATA  = "https://api.gdc.cancer.gov/data"
SSM_OCC = "https://api.gdc.cancer.gov/ssm_occurrences"

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

def _request_with_retries(method: str, url: str, *, json_body=None, timeout: int = 120, max_retries: int = 5, stream: bool = False) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            if method == "POST":
                r = requests.post(url, json=json_body, timeout=timeout, stream=stream)
            else:
                r = requests.get(url, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
    raise last_exc

def gdc_post_all(endpoint: str, payload: Dict[str, Any], page_size: int, timeout: int = 120) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    offset = 0
    while True:
        body = dict(payload)
        body["size"] = page_size
        body["from"] = offset
        r = _request_with_retries("POST", endpoint, json_body=body, timeout=timeout)
        data = r.json().get("data", {})
        hits = data.get("hits", [])
        if not hits:
            break
        results.extend(hits)
        offset += page_size
    return results

def fetch_case_ids_with(file_filter: Dict[str, Any], *, program: str, project_id: Optional[str], page_size: int, timeout: int = 120) -> Set[str]:
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
    for f in gdc_post_all(FILES, payload, page_size, timeout=timeout):
        for c in f.get("cases", []) or []:
            cid = c.get("case_id")
            if cid:
                case_ids.add(cid)
    return case_ids

def files_for_case(case_id: str, extra_filters: List[Dict[str, Any]], page_size: int, timeout: int = 120) -> List[Dict[str, Any]]:
    fields = ",".join([
        "file_id","file_name","md5sum","data_category","data_type","data_format",
        "experimental_strategy","workflow_type",
        "cases.submitter_id","cases.project.project_id",
        "cases.samples.submitter_id","cases.samples.sample_type",
        # slide QC nested under samples.portions.slides
        "cases.samples.portions.slides.percent_tumor_cells",
        "cases.samples.portions.slides.percent_stromal_cells",
        "cases.samples.portions.slides.percent_tumor_nuclei",
        "cases.samples.portions.slides.section_location"
    ])
    content = [{"op": "in", "content": {"field": "cases.case_id", "value": [case_id]}}] + extra_filters
    payload = {"filters": {"op": "and", "content": content}, "fields": fields}
    return gdc_post_all(FILES, payload, page_size, timeout=timeout)

def fetch_case_labels(case_id: str, timeout: int = 120) -> Dict[str, Any]:
    # Include diagnoses, exposures, survival-ish, slides
    fields = ",".join([
        "case_id","submitter_id","project.project_id","primary_site","disease_type",
        # Diagnoses
        "diagnoses.primary_diagnosis","diagnoses.morphology",
        "diagnoses.tumor_stage","diagnoses.ajcc_pathologic_stage",
        "diagnoses.classification_of_tumor","diagnoses.tumor_grade",
        "diagnoses.days_to_death","diagnoses.days_to_last_follow_up",
        # Demographic vital
        "demographic.vital_status",
        # Exposures
        "exposures.cigarettes_per_day","exposures.years_smoked","exposures.pack_years_smoked",
        # Slides QC by sample
        "samples.submitter_id","samples.sample_type",
        "samples.portions.slides.percent_tumor_cells",
        "samples.portions.slides.percent_stromal_cells",
        "samples.portions.slides.percent_tumor_nuclei",
        "samples.portions.slides.section_location"
    ])
    payload = {"filters": {"op": "in", "content": {"field": "case_id", "value": [case_id]}},
               "fields": fields, "size": 1}
    r = _request_with_retries("POST", CASES, json_body=payload, timeout=timeout)
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
                    "workflow_type": meta.get("workflow_type"),
                    # slide QC extract for convenience
                    "slide_qc": _extract_slide_qc_from_file_record(s)
                }
                if grouped[sid]["sample_type"] is None and stype:
                    grouped[sid]["sample_type"] = stype
                grouped[sid][key].append(item)
                attached = True
    # Fallback sample id guess from slide file name if needed
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
                "workflow_type": meta.get("workflow_type"),
                "slide_qc": None
            }
            grouped[sid][key].append(item)

def _local_name_from_filename(file_name: str) -> str:
    # For .gz but not .tar.gz, predict post-decompression filename
    if file_name.endswith(".gz") and not file_name.endswith(".tar.gz"):
        return file_name[:-3]
    return file_name

def _extract_slide_qc_from_file_record(sample_node: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    # sample_node is a cases.samples node from a files hit
    portions = sample_node.get("portions") or []
    ptc = psc = ptn = None
    sloc = None
    for p in portions:
        for sl in p.get("slides", []) or []:
            # take first non null, or keep max tumor percent if multiple
            if sl.get("percent_tumor_cells") is not None:
                ptc = max(ptc, sl.get("percent_tumor_cells")) if ptc is not None else sl.get("percent_tumor_cells")
            if sl.get("percent_stromal_cells") is not None:
                psc = max(psc, sl.get("percent_stromal_cells")) if psc is not None else sl.get("percent_stromal_cells")
            if sl.get("percent_tumor_nuclei") is not None:
                ptn = max(ptn, sl.get("percent_tumor_nuclei")) if ptn is not None else sl.get("percent_tumor_nuclei")
            if sl.get("section_location") and sloc is None:
                sloc = sl.get("section_location")
    if ptc is None and psc is None and ptn is None and sloc is None:
        return None
    return {
        "percent_tumor_cells": ptc,
        "percent_stromal_cells": psc,
        "percent_tumor_nuclei": ptn,
        "section_location": sloc
    }

def _patient_barcode_from_sample(sample_submitter_id: str) -> str:
    # TCGA-XX-YYYY-... -> TCGA-XX-YYYY
    parts = (sample_submitter_id or "").split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else sample_submitter_id

# -----------------------------
# BRCA receptor status from BCR Biotab
# -----------------------------
def _download_text_file(file_id: str, timeout: int = 120) -> str:
    r = _request_with_retries("GET", f"{DATA}/{file_id}", timeout=timeout, stream=True)
    # Content is the file itself. Read into bytes then decode
    content = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1<<20):
        if chunk:
            content.write(chunk)
    # Try utf-8 then latin-1 as fallback
    try:
        return content.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return content.getvalue().decode("latin-1")

def load_bcr_biotab_receptors(project_id: str, page_size: int, timeout: int = 120, debug: bool = False) -> Dict[str, Dict[str, Optional[str]]]:
    """
    For TCGA-BRCA only, fetch clinical_patient_brca.txt (BCR Biotab) and parse ER/PR/HER2.
    Returns mapping: patient_barcode -> {er_status_by_ihc, pr_status_by_ihc, her2_status_by_ihc, triple_negative}
    """
    if project_id != "TCGA-BRCA":
        return {}

    file_candidates = ["clinical_patient_brca.txt", "nationwidechildrens.org_clinical_patient_brca.txt"]
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Clinical"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["BCR Biotab"]}},
            {"op": "in", "content": {"field": "file_name", "value": file_candidates}}
        ]
    }
    payload = {"filters": filters, "fields": "file_id,file_name"}
    hits = gdc_post_all(FILES, payload, page_size, timeout=timeout)

    pick = None
    for h in hits:
        fn = h.get("file_name", "")
        if fn in file_candidates:
            pick = h
            break
    if pick is None and hits:
        pick = hits[0]

    if pick is None:
        if debug:
            print("[DEBUG] BRCA Biotab not found under Clinical Supplement BCR Biotab. Filters used:")
            print(json.dumps(filters, indent=2))
        return {}

    text = _download_text_file(pick["file_id"], timeout=timeout)
    # Parse TSV with tab delimiter
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for row in reader:
        pid = row.get("bcr_patient_barcode")
        if not pid:
            continue
        er = row.get("er_status_by_ihc")
        pr = row.get("pr_status_by_ihc")
        her2 = row.get("her2_status_by_ihc")
        triple = None
        if er is not None or pr is not None or her2 is not None:
            triple = (
                (er or "").strip().lower() == "negative" and
                (pr or "").strip().lower() == "negative" and
                (her2 or "").strip().lower() == "negative"
            )
        out[pid] = {
            "er_status_by_ihc": er if er else None,
            "pr_status_by_ihc": pr if pr else None,
            "her2_status_by_ihc": her2 if her2 else None,
            "triple_negative": triple
        }
    if debug:
        print(f"[DEBUG] Parsed BRCA Biotab receptors for {len(out)} patients from {pick.get('file_name')}")
    return out

# -----------------------------
# Mutations via /analysis/ssm_occurrences
# -----------------------------
def default_gene_panel_for_project(project_id: str) -> List[str]:
    pj = (project_id or "").upper()
    if pj == "TCGA-BRCA":
        return ["TP53","PIK3CA","GATA3","MAP3K1","CDH1","PTEN","AKT1","MAP2K4","RB1","ERBB2"]
    if pj == "TCGA-LUAD":
        return ["EGFR","KRAS","STK11","KEAP1","TP53","ALK","BRAF","ERBB2","MET","ROS1"]
    # reasonable pan-cancer subset often used in slide-genomics studies
    return ["TP53","PIK3CA","PTEN","RB1","BRAF","KRAS","EGFR","NF1","IDH1","ERBB2"]

def fetch_mutation_presence(
    case_ids: List[str],
    gene_panel: List[str],
    page_size: int,
    timeout: int = 120,
    debug: bool = False
) -> Dict[str, Dict[str, bool]]:
    """
    Determine per-case mutation presence for a gene panel using /ssm_occurrences.
    Returns: case_id -> {gene_symbol: True/False}.
    """
    if not case_ids or not gene_panel:
        return {}

    # Filter by cases and your panel
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "case.case_id", "value": case_ids}},
            {"op": "in", "content": {"field": "ssm.consequence.transcript.gene.symbol", "value": gene_panel}},
        ],
    }

    # Ask for only what we need
    payload = {
        "filters": filters,
        "fields": "case.case_id,ssm.ssm_id,ssm.consequence.transcript.gene.symbol",
    }

    hits = gdc_post_all(SSM_OCC, payload, page_size, timeout=timeout)

    # Initialize flags to False
    out: Dict[str, Dict[str, bool]] = {cid: {g: False for g in gene_panel} for cid in case_ids}

    # Track unique (case, gene, ssm) to avoid double counting
    seen: Set[Tuple[str, str, Optional[str]]] = set()

    for h in hits:
        cid = ((h.get("case") or {}).get("case_id")) or h.get("case_id")
        ssm = h.get("ssm") or {}
        ssm_id = ssm.get("ssm_id") or h.get("ssm_id")
        conseq = ssm.get("consequence")

        # Collect gene symbols from the consequence field, which can be list or dict
        symbols: List[str] = []
        if isinstance(conseq, list):
            for c in conseq:
                tr = (c or {}).get("transcript") or {}
                g = (tr.get("gene") or {}).get("symbol")
                if g:
                    symbols.append(g)
        elif isinstance(conseq, dict):
            tr = (conseq.get("transcript") or {})
            g = (tr.get("gene") or {}).get("symbol")
            if g:
                symbols.append(g)
        else:
            # Rare fallback if API returns a flat 'genes' list
            symbols = [g.get("symbol") for g in (h.get("genes") or []) if g.get("symbol")]

        # Deduplicate by case, gene, and ssm id
        for sym in set(symbols):
            if cid and sym and sym in out.get(cid, {}):
                key = (cid, sym, ssm_id)
                if key in seen:
                    continue
                seen.add(key)
                out[cid][sym] = True

    if debug:
        pos = sum(sum(1 for v in d.values() if v) for d in out.values())
        print(f"[mut] cases={len(case_ids)} genes={len(gene_panel)} hits={len(hits)} positives={pos}")
        # Show one positive example if present
        for cid, flags in out.items():
            genes_pos = [g for g, v in flags.items() if v]
            if genes_pos:
                print(f"[mut] example {cid}: {genes_pos[:8]}")
                break
        if pos == 0:
            print("[mut][warn] All False. Dumping payload for inspection:")
            print(json.dumps(payload, indent=2))

    return out

# -----------------------------
# Build per-case records
# -----------------------------
def build_records_for_case(case_id: str, page_size: int, receptor_map: Dict[str, Dict[str, Optional[str]]],
                           mut_index: Dict[str, Dict[str, bool]], timeout: int = 120) -> List[Dict[str, Any]]:
    # Files
    slide_meta = files_for_case(case_id, [
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}}
    ], page_size, timeout=timeout)
    beta_meta = files_for_case(case_id, [
        {"op": "and", "content": [
            {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}}
        ]}
    ], page_size, timeout=timeout)

    grouped = defaultdict(lambda: {"sample_type": None, "slides": [], "betas": []})
    for m in slide_meta:
        _attach(grouped, m, "slides")
    for m in beta_meta:
        _attach(grouped, m, "betas")

    candidates = {sid: grp for sid, grp in grouped.items() if grp["slides"] and grp["betas"]}
    if not candidates:
        return []

    # Clinical labels
    case_meta = fetch_case_labels(case_id, timeout=timeout)
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
        "classification_of_tumor": primary_dx.get("classification_of_tumor"),
        "tumor_grade": primary_dx.get("tumor_grade")
    }

    # Survival
    vital = (case_meta.get("demographic") or {}).get("vital_status")
    dtd = primary_dx.get("days_to_death")
    dtlfu = primary_dx.get("days_to_last_follow_up")
    os_days = None
    event = None
    if vital:
        vs = vital.lower()
        if vs == "dead":
            event = 1
            os_days = dtd if dtd is not None else None
        elif vs == "alive":
            event = 0
            os_days = dtlfu if dtlfu is not None else None

    # Exposures
    exp = (case_meta.get("exposures") or [])
    exposure_pick = exp[0] if exp else {}
    cigs_per_day = exposure_pick.get("cigarettes_per_day")
    years_smoked = exposure_pick.get("years_smoked")
    pack_years_smoked = exposure_pick.get("pack_years_smoked")
    computed_pack_years = None
    try:
        if cigs_per_day is not None and years_smoked is not None:
            computed_pack_years = float(cigs_per_day) / 20.0 * float(years_smoked)
    except Exception:
        computed_pack_years = None

    # Slide QC by sample_submitter_id
    sample_slide_qc: Dict[str, Dict[str, Optional[float]]] = {}
    for s in (case_meta.get("samples") or []):
        sid = s.get("submitter_id")
        if not sid:
            continue
        portions = s.get("portions") or []
        ptc = psc = ptn = None
        sloc = None
        for p in portions:
            for sl in p.get("slides", []) or []:
                if sl.get("percent_tumor_cells") is not None:
                    ptc = max(ptc, sl.get("percent_tumor_cells")) if ptc is not None else sl.get("percent_tumor_cells")
                if sl.get("percent_stromal_cells") is not None:
                    psc = max(psc, sl.get("percent_stromal_cells")) if psc is not None else sl.get("percent_stromal_cells")
                if sl.get("percent_tumor_nuclei") is not None:
                    ptn = max(ptn, sl.get("percent_tumor_nuclei")) if ptn is not None else sl.get("percent_tumor_nuclei")
                if sl.get("section_location") and sloc is None:
                    sloc = sl.get("section_location")
        if any(v is not None for v in (ptc, psc, ptn, sloc)):
            sample_slide_qc[sid] = {
                "percent_tumor_cells": ptc,
                "percent_stromal_cells": psc,
                "percent_tumor_nuclei": ptn,
                "section_location": sloc
            }

    # Mutation flags for this case_id
    mutations_for_case = mut_index.get(case_id) or {}

    records: List[Dict[str, Any]] = []
    for sample_id, grp in candidates.items():
        slide = sort_files_first(grp["slides"])[0]
        beta  = sort_files_first(grp["betas"])[0]

        slide_name = slide.get("file_name") or slide["file_id"]
        beta_name  = beta.get("file_name") or beta["file_id"]
        slide_local = f"{sample_id}/{_local_name_from_filename(slide_name)}"
        beta_local  = f"{sample_id}/{_local_name_from_filename(beta_name)}"

        slide_out = dict(slide)
        slide_out["local_path"] = slide_local
        # blend in slide QC from case-level for this sample if missing
        if slide_out.get("slide_qc") is None and sample_id in sample_slide_qc:
            slide_out["slide_qc"] = sample_slide_qc[sample_id]

        beta_out = dict(beta)
        beta_out["local_path"] = beta_local

        # BRCA receptors, patient-level
        patient_barcode = _patient_barcode_from_sample(sample_id)
        brca_rec = receptor_map.get(patient_barcode) if receptor_map else None

        # Compose labels block
        labels = {
            "wsi_slide": {
                "morphology": dx.get("morphology"),
                "tumor_grade": dx.get("tumor_grade"),
                "tumor_stage": dx.get("tumor_stage"),
                "ajcc_pathologic_stage": dx.get("ajcc_pathologic_stage"),
                "slide_qc": slide_out.get("slide_qc")  # dict or None
            },
            "brca_receptors": brca_rec,  # None for non-BRCA or missing
            "mutations": mutations_for_case if mutations_for_case else None,
            "survival": {
                "vital_status": vital,
                "event": event,
                "os_days": os_days,
                "days_to_last_follow_up": dtlfu,
                "days_to_death": dtd
            },
            "exposure": {
                "cigarettes_per_day": cigs_per_day,
                "years_smoked": years_smoked,
                "pack_years_smoked": pack_years_smoked,
                "computed_pack_years": computed_pack_years
            }
        }

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
            "methylation_beta": beta_out,
            "labels": labels
        }
        records.append(rec)

    return records

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build paired WSI + DNA methylation index and add downstream labels.")
    parser.add_argument("--project_id", type=str, default="TCGA-BRCA", help="TCGA project id or None for all TCGA")
    parser.add_argument("--program", type=str, default="TCGA", help="Program filter")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional limit on number of paired cases")
    parser.add_argument("--api_page_size", type=int, default=1000)
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction for split")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for splitting")
    parser.add_argument("--split_unit", type=str, choices=["case","sample"], default="case", help="Split unit")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    parser.add_argument("--debug", action="store_false", help="Print extra debugging info for clinical and mutation lookups")
    parser.add_argument("--mut_genes", type=str, default="", help="Comma separated custom gene panel to query in ssm_occurrences")
    args = parser.parse_args()

    data_root = project_root / "data/wsi_dnameth"
    train_dir = data_root / "train"
    test_dir  = data_root / "test"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    # Find paired cases
    slide_cases = fetch_case_ids_with(
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        program=args.program, project_id=args.project_id, page_size=args.api_page_size, timeout=args.timeout
    )
    beta_cases = fetch_case_ids_with(
        {"op": "and", "content": [
            {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}}
        ]},
        program=args.program, project_id=args.project_id, page_size=args.api_page_size, timeout=args.timeout
    )
    paired_cases = sorted(slide_cases & beta_cases)
    if args.max_cases is not None:
        paired_cases = paired_cases[:args.max_cases]

    print(f"Found {len(paired_cases)} paired cases" + (f" in project {args.project_id}" if args.project_id else ""))

    # Prepare downstream label sources
    receptor_map: Dict[str, Dict[str, Optional[str]]] = load_bcr_biotab_receptors(
        args.project_id, page_size=args.api_page_size, timeout=args.timeout, debug=args.debug
    )

    # Mutation panel
    if args.mut_genes.strip():
        gene_panel = [g.strip().upper() for g in args.mut_genes.split(",") if g.strip()]
    else:
        gene_panel = default_gene_panel_for_project(args.project_id)

    mut_index: Dict[str, Dict[str, bool]] = {}
    if paired_cases and gene_panel:
        mut_index = fetch_mutation_presence(paired_cases, gene_panel, page_size=args.api_page_size, timeout=args.timeout, debug=args.debug)

    # Build full record list
    full_records: List[Dict[str, Any]] = []
    for i, cid in enumerate(paired_cases, 1):
        print(f"[{i}/{len(paired_cases)}] case {cid}")
        recs = build_records_for_case(cid, page_size=args.api_page_size, receptor_map=receptor_map, mut_index=mut_index, timeout=args.timeout)
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
