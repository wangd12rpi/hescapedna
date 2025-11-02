# hescape/models/dnameth_models/_cpgpt.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

# CpGPT imports (use your clone; ensure it is importable or on PYTHONPATH)
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer

from hescape.models._cache import EmbeddingCache

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


def _unique_linear_leaf_names(module: torch.nn.Module) -> List[str]:
    """
    Collect unique leaf names of nn.Linear modules for robust PEFT LoRA targeting
    (uses endswith matching in PEFT).
    """

    targets = ["linear1"]
    return targets


class CpGPTRunner:
    """
    Thin wrapper around your working CpGPT pipeline.

    Usage:
      runner = CpGPTRunner(root="/path/cpgpt_files", enable_lora=True)
      embeds = runner.encode_beta_files_autograd([path1, path2, ...])  # torch.FloatTensor [B, D] w/ grads
    """

    def __init__(
        self,
        root: str,
        model_name: str = "cancer",
        device: str | None = None,
        precision: str = "16-mixed",
        cache_embeddings: bool = True,
        enable_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_targets: List[str] | None = None,
    ) -> None:
        model_name = "large"  # enforce for now unless you change config layout
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
        # do NOT download anything; you already have the files
        self.config = self.inferencer.load_cpgpt_config(self.model_cfg)
        self.model = self.inferencer.load_cpgpt_model(self.config, model_ckpt_path=self.model_ckpt, strict_load=True)

        # grab DNA-LLM settings from config
        self.dna_llm = self.config.data.dna_llm
        self.dna_context_len = self.config.data.dna_context_len
        self.sorting_strategy = self.config.data.sorting_strategy
        self.embedding_dim = int(getattr(self.config.model.net, "d_embedding", 128))

        # Lightning trainer used only for the legacy frozen fast path
        self.trainer = CpGPTTrainer(precision=precision)

        # read vocab and pin the input site list
        self.vocab_sites = self._load_vocab_sites(self.model_vocab)

        # device hint
        self.device = device

        # Initialize cache for per-file embeddings
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = EmbeddingCache("cpgpt_embeddings") if cache_embeddings else None

        # LoRA setup (base frozen)
        self.enable_lora = enable_lora
        if enable_lora:
            if not _PEFT_AVAILABLE:
                raise RuntimeError("peft is required for CpGPT LoRA finetuning but is not installed.")
            targets = lora_targets or _unique_linear_leaf_names(self.model)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    target_modules=targets,
                    bias="none",
                ),
            )
            # By default, PEFT sets only adapter params as trainable; base remains frozen.

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

    # -------------------------------
    # Batch preprocessing utilities
    # -------------------------------
    def _prepare_processed_dir(self, txt_paths: List[str]) -> str:
        """
        Convert a batch of raw TSVs into CpGPT processed directory usable by CpGPTDataModule.
        Returns the path to the processed directory.
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
        tmp_root = tempfile.TemporaryDirectory(dir=self.data_dir)
        feather_path = os.path.join(tmp_root.name, "filtered_batch.feather")
        combined.to_feather(feather_path)

        processed_dir = os.path.join(tmp_root.name, "processed")
        Path(processed_dir).mkdir(parents=True, exist_ok=True)

        # Save to CpGPT processed format
        datasaver = CpGPTDataSaver(data_paths=feather_path, processed_dir=processed_dir)
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber

        embedder = DNALLMEmbedder(dependencies_dir=self.human_dir)
        prober = IlluminaMethylationProber(dependencies_dir=self.human_dir, embedder=embedder)
        datasaver.process_files(prober, embedder)

        # Hold onto the tempdir so it isn't GC'd before dataloading
        self._last_tmp_for_batch = tmp_root
        return processed_dir

    # -------------------------------
    # Frozen fast path (cached)
    # -------------------------------
    def encode_beta_files(self, txt_paths: List[str]) -> torch.Tensor:
        """
        Encode a batch of beta text files into CpGPT embeddings with per-file caching.
        Each file must be a 2-column TSV: CpG_Site, Beta_Value
        Returns: torch.FloatTensor [B, D] (typically D=128) on CPU.
        """
        assert len(txt_paths) > 0, "encode_beta_files needs at least one path"

        if not self.cache_embeddings or self.embedding_cache is None:
            return self._compute_embeddings(txt_paths).to(dtype=torch.float32)

        cached_embeddings: Dict[str, torch.Tensor] = {}
        uncached_paths: List[str] = []

        for path in txt_paths:
            cached = self.embedding_cache.get(path)
            if cached is not None:
                cached_embeddings[path] = cached
            else:
                uncached_paths.append(path)

        if uncached_paths:
            uncached_embeds = self._compute_embeddings(uncached_paths)
            for path, emb in zip(uncached_paths, uncached_embeds):
                emb_single = emb.unsqueeze(0) if emb.dim() == 1 else emb
                self.embedding_cache.set(path, emb_single)
                cached_embeddings[path] = emb_single

        result = [cached_embeddings[path].squeeze(0) for path in txt_paths]
        return torch.stack(result)

    def _compute_embeddings(self, txt_paths: List[str]) -> torch.Tensor:
        """
        Internal method to compute embeddings for a batch of files (frozen path).
        Uses Lightning Trainer.predict (no grad).
        """
        processed_dir = self._prepare_processed_dir(txt_paths)
        datamodule = CpGPTDataModule(
            predict_dir=processed_dir,
            dependencies_dir=self.human_dir,
            batch_size=len(txt_paths),
            num_workers=0,
            max_length=20_000,  # tutorial default
            dna_llm=self.dna_llm,
            dna_context_len=self.dna_context_len,
            sorting_strategy=self.sorting_strategy,
            pin_memory=False,
        )

        out = self.trainer.predict(
            model=self.model,
            datamodule=datamodule,
            predict_mode="forward",
            return_keys=["sample_embedding"],
        )
        # out is a dict with "sample_embedding": torch.Tensor [B, D]
        emb = out.get("sample_embedding", None)
        if emb is None:
            # be robust to plural naming used in some versions
            emb = out.get("sample_embeddings")
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)

        return emb.to(dtype=torch.float32)  # [B, D]

    # -------------------------------
    # Autograd path (for LoRA finetuning)
    # -------------------------------
    def encode_beta_files_autograd(self, txt_paths: List[str]) -> torch.Tensor:
        """
        Encode beta files into CpGPT embeddings **with gradients enabled** so that LoRA adapters can train.

        Returns: torch.FloatTensor [B, D] on the model's device.
        """
        assert len(txt_paths) > 0, "encode_beta_files_autograd needs at least one path"
        processed_dir = self._prepare_processed_dir(txt_paths)

        datamodule = CpGPTDataModule(
            predict_dir=processed_dir,
            dependencies_dir=self.human_dir,
            batch_size=len(txt_paths),
            num_workers=0,
            max_length=20_000,
            dna_llm=self.dna_llm,
            dna_context_len=self.dna_context_len,
            sorting_strategy=self.sorting_strategy,
            pin_memory=False,
        )
        # Ask the datamodule for predict loader
        try:
            datamodule.setup(stage="predict")
        except Exception:
            # many Lightning datamodules also work without explicit setup
            pass

        loader = datamodule.predict_dataloader()
        device = next(self.model.parameters()).device
        self.model.train()  # train mode to enable train-time layers for finetuning

        # Mirror CpGPTTrainer.predict's attribute passing convention
        setattr(self.model, "predict_mode_predict", "forward")
        setattr(self.model, "return_keys_predict", ["sample_embedding"])

        outputs: List[torch.Tensor] = []
        # Iterate and directly call predict_step **with grad enabled**
        for batch_idx, batch in enumerate(loader):
            # Move batch tensors to device when applicable
            if isinstance(batch, dict):
                for k, v in list(batch.items()):
                    if torch.is_tensor(v):
                        batch[k] = v.to(device)
            # The CpGPT LightningModule implements predict_step and returns a dict
            pred = self.model.predict_step(batch, batch_idx=batch_idx)
            emb = pred.get("sample_embedding", None)
            if emb is None:
                emb = pred.get("sample_embeddings")
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, device=device)
            outputs.append(emb)

        # Clean up the temporary attributes
        delattr(self.model, "predict_mode_predict")
        delattr(self.model, "return_keys_predict")

        if not outputs:
            # no data?
            return torch.empty((0, int(self.embedding_dim)), device=device, dtype=torch.float32)
        emb = torch.cat(outputs, dim=0).to(dtype=torch.float32)
        return emb


def _build_cpgpt_model(
    checkpoint_root: str | Path,
    in_features: int,
    out_features: int,
    *,
    model_name: str = "cancer",
    cache_embeddings: bool = True,
    enable_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_targets: List[str] | None = None,
    **kwargs: Any,
) -> CpGPTRunner:
    """
    Factory to match hescape's encoder builder pattern.
    Returns a CpGPTRunner instance. Projection heads are handled by the caller.
    """
    runner = CpGPTRunner(
        root=str(checkpoint_root),
        model_name=model_name,
        cache_embeddings=cache_embeddings,
        enable_lora=enable_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )
    return runner
