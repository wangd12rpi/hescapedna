# hescape/models/dnameth_models/_cpgpt.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# CpGPT imports (use your clone; ensure it is importable or on PYTHONPATH)
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

from hescape.models._cache import EmbeddingCache


class CpGPTRunner:
    """
    Thin wrapper around your working CpGPT pipeline.

    Usage:
      runner = CpGPTRunner(root="/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files")
      embeds = runner.encode_beta_files([path1, path2, ...])  # torch.FloatTensor [B, 128]
    """

    def __init__(
        self,
        root: str,
        model_name: str = "cancer",
        device: str | None = None,
        precision: str = "16-mixed",
    ) -> None:
        model_name = 'cancer'
        self.root = Path(root).resolve()
        self.dependencies_dir = str(self.root / "dependencies")
        self.human_dir = str(self.root / "dependencies" / "human")
        self.data_dir = str(self.root / "data")

        self.model_name = model_name
        self.model_ckpt = str(self.root / "dependencies" / "model" / "weights" / f"{model_name}.ckpt")
        self.model_cfg = str(self.root / "dependencies" / "model" / "config" / f"{model_name}.yaml")
        self.model_vocab = str(self.root / "dependencies" / "model" / "vocab" / f"{model_name}.json")

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dependencies_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.data_dir, "raw")).mkdir(parents=True, exist_ok=True)

        # load once
        self.inferencer = CpGPTInferencer(dependencies_dir=self.dependencies_dir, data_dir=self.data_dir)

        print(self.model_cfg)
        # do NOT download anything; you already have the files
        self.config = self.inferencer.load_cpgpt_config(self.model_cfg)
        self.model = self.inferencer.load_cpgpt_model(self.config, model_ckpt_path=self.model_ckpt, strict_load=True)

        # grab DNA-LLM settings from config
        self.dna_llm = self.config.data.dna_llm
        self.dna_context_len = self.config.data.dna_context_len
        self.sorting_strategy = self.config.data.sorting_strategy
        self.embedding_dim = int(getattr(self.config.model.net, "d_embedding", 128))

        # trainer used for predict
        self.trainer = CpGPTTrainer(precision=precision)

        # read vocab and pin the input site list
        self.vocab_sites = self._load_vocab_sites(self.model_vocab)

        # device hint only affects how Lightning places the model; no manual .to()
        self.device = device

        # Initialize cache for per-file embeddings
        self.embedding_cache = EmbeddingCache('cpgpt_embeddings')

    @staticmethod
    def _load_vocab_sites(vocab_json: str) -> List[str]:
        """
        CpGPT cancer vocab commonly stores CpG ids under key 'input'.
        Fallbacks provided for robustness.
        """
        with open(vocab_json, "r") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return [str(s) for s in obj]

        for k in ("input", "sites", "var_names", "features"):
            if k in obj and isinstance(obj[k], list):
                return [str(s) for s in obj[k]]

        raise ValueError(f"Unrecognized vocab schema in {vocab_json}")

    def encode_beta_files(self, txt_paths: List[str]) -> torch.Tensor:
        """
        Encode a batch of beta text files into CpGPT embeddings with per-file caching.

        Each file must be a 2-column TSV: CpG_Site, Beta_Value
        Returns:
          torch.FloatTensor [B, D] typically D=128 for the 'cancer' model
        """
        assert len(txt_paths) > 0, "encode_beta_files needs at least one path"

        # Check cache for each file individually
        cached_embeddings = {}
        uncached_paths = []

        for path in txt_paths:
            cached = self.embedding_cache.get(path)
            if cached is not None:
                cached_embeddings[path] = cached
            else:
                uncached_paths.append(path)

        # Compute embeddings for uncached files (batch them for efficiency)
        if uncached_paths:
            print(uncached_paths)
            uncached_embeds = self._compute_embeddings(uncached_paths)
            # Cache each file individually
            for path, emb in zip(uncached_paths, uncached_embeds):
                # Store as [1, D] tensor for consistency
                emb_single = emb.unsqueeze(0) if emb.dim() == 1 else emb
                self.embedding_cache.set(path, emb_single)
                cached_embeddings[path] = emb_single

        # Reconstruct batch in original order
        result = [cached_embeddings[path].squeeze(0) for path in txt_paths]
        return torch.stack(result)

    def _compute_embeddings(self, txt_paths: List[str]) -> torch.Tensor:
        """
        Internal method to compute embeddings for a batch of files.
        Extracted from encode_beta_files for caching purposes.
        """
        # Build a combined table (rows = samples, columns = CpG sites)
        rows = []
        for i, p in enumerate(txt_paths):
            df = pd.read_csv(p, sep="\t", header=None, names=["CpG_Site", "Beta_Value"])
            df["Beta_Value"] = pd.to_numeric(df["Beta_Value"], errors="coerce")
            m = dict(zip(df["CpG_Site"], df["Beta_Value"]))
            row = {"CASE_ID": str(i)}
            # keep only CpGs in model vocab and in the *vocab order*
            for cpg in self.vocab_sites:
                row[cpg] = m.get(cpg, None)
            rows.append(row)

        combined = pd.DataFrame(rows)  # shape [B, 1 + N_vocab]
        with tempfile.TemporaryDirectory(dir=self.data_dir) as tmp_root:
            feather_path = os.path.join(tmp_root, "filtered_batch.feather")
            combined.to_feather(feather_path)

            processed_dir = os.path.join(tmp_root, "processed")
            Path(processed_dir).mkdir(parents=True, exist_ok=True)

            # Save to CpGPT processed format
            datasaver = CpGPTDataSaver(data_paths=feather_path, processed_dir=processed_dir)
            # embedder + prober are constructed internally by DataSaver through inputs passed below.
            from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
            from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber

            embedder = DNALLMEmbedder(dependencies_dir=self.human_dir)
            prober = IlluminaMethylationProber(dependencies_dir=self.human_dir, embedder=embedder)
            datasaver.process_files(prober, embedder)

            # Build datamodule for predict
            datamodule = CpGPTDataModule(
                predict_dir=processed_dir,
                dependencies_dir=self.human_dir,
                batch_size=len(txt_paths),
                num_workers=0,
                max_length=20_000,  # default tailored for tutorial
                dna_llm=self.dna_llm,
                dna_context_len=self.dna_context_len,
                sorting_strategy=self.sorting_strategy,
                pin_memory=False,
            )

            # Predict embeddings
            out = self.trainer.predict(
                model=self.model,
                datamodule=datamodule,
                predict_mode="forward",
                return_keys=["sample_embedding"],
            )
        # out is a dict with "sample_embedding": torch.Tensor [B, D]
        emb = out["sample_embedding"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)

        return emb.to(dtype=torch.float32)  # [B, D]


def _build_cpgpt_model(
    checkpoint_root: str | Path,
    in_features: int,   # not used here; kept for API parity with encoders
    out_features: int,  # desired head dim in the calling encoder
    **kwargs,
) -> CpGPTRunner:
    """
    Factory to match hescape's encoder builder pattern.
    Returns a CpGPTRunner instance. Projection heads are handled by the caller.
    """
    runner = CpGPTRunner(root=str(checkpoint_root), model_name=kwargs.get("model_name", "cancer"))
    return runner
