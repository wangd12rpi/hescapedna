from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---- CpGPT imports (repo must be importable) ----
# Minimal, battle-tested imports based on the repo layout you shared earlier.
# If your clone uses slightly different module names, update the import lines below.
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder  # builds DNA-LLM memmap index
from cpgpt.model.lit_module import CpGPTLitModule  # Lightning wrapper (we'll use its .net)
from cpgpt.model.build import build_net_from_config  # factory to create CpGPT net from YAML
from cpgpt.utils.io import load_yaml  # lightweight YAML loader


class CpGPTBackbone(nn.Module):
    """
    CpGPT backbone that turns a methylation beta vector into a CpGPT sample embedding.

    Expected input to forward:
      - x:  Float tensor [B, N_vocab] with beta values aligned to CpGPT vocab order.
            NaNs are allowed for missing sites.
    Returns:
      - embedding: Float tensor [B, embed_dim] (e.g., 128 for the 'cancer' model)
    """

    def __init__(
        self,
        dependencies_root: str,
        model_name: str = "cancer",
        species_dir: str = "human",                # folder name under dependencies/
        species_key: str = "homo_sapiens",         # key used by DNALLMEmbedder
        dna_llm: str = "nucleotide-transformer-v2-500m-multi-species",
        dna_context_len: int = 2001,
        finetune_lora: bool | Dict[str, Any] = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Args:
          dependencies_root: root folder that contains 'dependencies/' from your tree.
            Example: '/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files'
          model_name: subfolder under dependencies/model/{config|weights|vocab}
          species_dir: folder under dependencies/ with species-specific DBs and memmaps
          species_key: key used inside CpGPT's metadata dict, typically 'homo_sapiens'
          dna_llm: DNA LLM id matching your memmap path
          dna_context_len: context length (must match memmap filename)
          finetune_lora: False or a dict with LoRA settings {'r': 8, 'alpha': 16, 'dropout': 0.0}
        """
        super().__init__()

        # ---- paths and sanity ----
        dep_root = Path(dependencies_root).resolve()
        self.paths = {
            "dep": dep_root / "dependencies",
            "model_cfg": dep_root / "dependencies" / "model" / "config" / f"{model_name}.yaml",
            "model_ckpt": dep_root / "dependencies" / "model" / "weights" / f"{model_name}.ckpt",
            "model_vocab": dep_root / "dependencies" / "model" / "vocab" / f"{model_name}.json",
            "dna_mmap": dep_root
            / "dependencies"
            / species_dir
            / "dna_embeddings"
            / species_key
            / dna_llm
            / f"{dna_context_len}bp_dna_embeddings.mmap",
        }
        for k, p in self.paths.items():
            if k in {"dep"}:
                continue
            if not p.exists():
                raise FileNotFoundError(f"[CpGPTBackbone] Missing required file for '{k}': {p}")

        # ---- load vocab (order of CpG probes the model expects) ----
        self.vocab_sites = self._load_vocab(self.paths["model_vocab"])
        self.n_vocab = len(self.vocab_sites)

        # ---- embedder for DNA sequence features and metadata dictionaries ----
        self.embedder = DNALLMEmbedder(dependencies_dir=str(self.paths["dep"]))
        self.species_key = species_key
        self.dna_llm = dna_llm
        self.dna_context_len = dna_context_len

        # Build mapping from 'chrom:pos' -> index within the memmap
        self._embedding_index_dict = self.embedder.ensembl_metadata_dict[self.species_key][self.dna_llm][
            self.dna_context_len
        ]
        self._reverse_vocab = self.embedder.ensembl_metadata_dict[self.species_key]["reverse_vocab"]
        self._llm_embed_dim = self.embedder.llm_embedding_size_dict[self.dna_llm]

        # Memory-map the DNA-LLM embeddings once
        self._dna_mmap = np.memmap(
            self.paths["dna_mmap"],
            dtype="float32",
            mode="r",
            shape=(len(self._embedding_index_dict), self._llm_embed_dim),
        )

        # ---- build CpGPT net and load weights ----
        cfg = load_yaml(str(self.paths["model_cfg"]))
        self.net = build_net_from_config(cfg)  # returns the CpGPT 'net' module
        # Lightning wrapper is used simply as a loader to align state_dict keys
        lit = CpGPTLitModule(training=cfg.get("training", {}), net=self.net, optimizer=torch.optim.Adam(self.net.parameters()))
        ckpt = torch.load(self.paths["model_ckpt"], map_location="cpu")
        lit.load_state_dict(ckpt["state_dict"], strict=False)  # loads into lit.net as needed
        self.net = lit.net  # grab the net back
        self.net.eval()  # default inference mode
        self.to(device)
        self.device = torch.device(device)

        # Optional LoRA
        self._lora_on = False
        if finetune_lora:
            self._apply_lora(self.net, finetune_lora if isinstance(finetune_lora, dict) else {})

        # Expose output embedding size for upstream code (e.g., your head choice)
        # CpGPT typically uses 128-dim sample embeddings for cancer weights
        # If the config differs, infer from a dummy linear probe on net.sample_embed_dim if available.
        self.embed_dim = getattr(self.net, "sample_embed_dim", 128)

    # ---------- public API ----------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, N_vocab] beta matrix aligned to vocab/cancer.json order. NaNs allowed.

        Returns:
          sample embeddings [B, embed_dim]
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B, N_vocab], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_vocab:
            raise ValueError(
                f"Input has {x.shape[1]} features, but CpGPT vocab expects {self.n_vocab}. "
                f"Make sure your datamodule filtered & ordered CpGs by vocab json."
            )

        x = x.to(self.device, dtype=torch.float32)
        mask_na = torch.isnan(x)  # [B, N]
        x = torch.nan_to_num(x, nan=0.0)

        # Static per-site genomic metadata prepared once
        chroms_idx, positions = self._get_chrom_pos_for_vocab()  # [N], [N] on CPU
        chroms = chroms_idx.unsqueeze(0).expand(x.size(0), -1).to(self.device)      # [B, N]
        positions = positions.unsqueeze(0).expand(x.size(0), -1).to(self.device)    # [B, N]

        # Build sequence embeddings for all vocab CpGs via memmap lookup
        seq_emb = self._get_dna_embeddings_for_vocab(batch_size=x.size(0))  # [B, N, E]

        # Call CpGPT's sample encoder
        # CpGPT's net expects keys: meth, sequence_embeddings, chroms, positions, mask_na
        out = self.net.encode_sample(
            meth=x,  # [B, N] (beta values)
            sequence_embeddings=seq_emb,  # [B, N, E]
            chroms=chroms,               # [B, N] (int32)
            positions=positions,         # [B, N] (int32)
            mask_na=mask_na,             # [B, N] (bool)
        )
        # 'out' is the sample embedding [B, D]
        return out

    # ---------- builders & helpers ----------

    @staticmethod
    def _load_vocab(vocab_json: Path) -> List[str]:
        with open(vocab_json, "r") as f:
            data = json.load(f)
        # Accept several common shapes: a flat list or {'sites': [...]} or {'var_names': [...]}
        if isinstance(data, list):
            return [str(s) for s in data]
        for key in ("sites", "cpg_ids", "features", "var_names"):
            if key in data and isinstance(data[key], (list, tuple)):
                return [str(s) for s in data[key]]
        raise ValueError(f"Unrecognized vocab JSON format: {vocab_json}")

    def _get_chrom_pos_for_vocab(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolves chromosome indices and 1-based genomic positions for each CpG in the vocab.
        Uses CpGPT's metadata DB via DNALLMEmbedder, which ships with probe->coord mappings.

        Returns:
          chroms_idx: int32 tensor [N_vocab]
          positions:  int32 tensor [N_vocab]
        """
        # CpGPT packs an illumina metadata DB and manifests under dependencies/{human}/...
        # DNALLMEmbedder has already loaded metadata dicts; we use its helper dicts:
        #   - ensembl_metadata_dict[species]['vocab']: {'1':0, '2':1, ..., 'X':22, ...}
        #   - illumina probe -> genomic coordinate mapping is also exposed in embedder.
        # The embedder exposes a dict: embedder.illumina_metadata_dict[species]['cpg_to_loc']
        # where each entry is (chrom_label, position). If your local clone uses a different key,
        # update here accordingly.

        # Try preferred attribute first, then fallback via manifests.
        if hasattr(self.embedder, "illumina_metadata_dict"):
            d = self.embedder.illumina_metadata_dict[self.species_key]
            if "cpg_to_loc" in d:
                cpg_to_loc = d["cpg_to_loc"]  # { 'cg00000029': ('16', 53468180), ... }
            else:
                raise RuntimeError(
                    "DNALLMEmbedder missing 'cpg_to_loc'. Update CpGPT or adjust mapping access here."
                )
        else:
            raise RuntimeError(
                "DNALLMEmbedder does not expose 'illumina_metadata_dict'. "
                "Update to the current CpGPT repo and ensure dependencies are complete."
            )

        chroms_idx = []
        positions = []
        vocab_map = self.embedder.ensembl_metadata_dict[self.species_key]["vocab"]  # {'1':0, '2':1, ...}
        for probe in self.vocab_sites:
            try:
                chrom_label, pos = cpg_to_loc[probe]
            except KeyError:
                # Unknown probe — create a dummy pad token by mapping to an impossible location 0
                # CpGPT handles mask_na per-site; we still need integer placeholders.
                chroms_idx.append(-1)
                positions.append(-1)
                continue
            # map chromosome string to int index
            if chrom_label.startswith("chr"):
                chrom_label = chrom_label.replace("chr", "")
            if chrom_label not in vocab_map:
                chroms_idx.append(-1)
                positions.append(-1)
                continue
            chroms_idx.append(int(vocab_map[chrom_label]))
            positions.append(int(pos))

        chroms_idx = torch.tensor(chroms_idx, dtype=torch.int32)
        positions = torch.tensor(positions, dtype=torch.int32)
        return chroms_idx, positions

    def _get_dna_embeddings_for_vocab(self, batch_size: int) -> torch.Tensor:
        """
        Looks up the DNA-LLM embedding vector for each CpG site in vocab, then repeats across batch.

        Returns:
          seq_emb: float32 tensor [B, N_vocab, E]
        """
        chroms_idx, positions = self._get_chrom_pos_for_vocab()
        # convert chrom int -> label with reverse_vocab, then build 'label:pos' strings used by index dict
        loc_keys: List[str] = []
        for c, p in zip(chroms_idx.tolist(), positions.tolist()):
            if c < 0 or p < 0:
                loc_keys.append(None)  # missing site — we'll fill zeros
            else:
                chrom_label = self._reverse_vocab[c]  # '1', '2', 'X', ...
                loc_keys.append(f"{chrom_label}:{p}")

        indices: List[int] = []
        for key in loc_keys:
            if key is None or key not in self._embedding_index_dict:
                indices.append(-1)
            else:
                indices.append(int(self._embedding_index_dict[key]))

        # Gather rows from memmap; fill zeros for missing indices
        embs = np.zeros((self.n_vocab, self._llm_embed_dim), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx >= 0:
                embs[i] = self._dna_mmap[idx]
        seq_emb = torch.from_numpy(embs).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, E]
        return seq_emb.to(self.device)

    def _apply_lora(self, module: nn.Module, cfg: Dict[str, Any]) -> None:
        """
        Injects LoRA adapters into CpGPT Linear layers.
        cfg keys: r, alpha, dropout, target_modules (optional, list of substrings)
        """
        from peft import LoraConfig, get_peft_model  # lazy import
        r = int(cfg.get("r", 8))
        alpha = int(cfg.get("alpha", 16))
        dropout = float(cfg.get("dropout", 0.0))
        target_modules: List[str] = cfg.get("target_modules", [])

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules if target_modules else None,
        )
        peft_model = get_peft_model(module, lora_cfg)
        self.net = peft_model
        self._lora_on = True


def _build_cpgpt_model(
    checkpoint_root: str | Path,
    in_features: int,    # unused by CpGPT; kept for API parity with your encoder wrapper
    out_features: int,   # desired projection size after CpGPT; you can keep CpGPT's native dim too
    **kwargs: Any,
) -> nn.Module:
    """
    Factory used by your `GexpEncoder` case 'cpgpt'.

    Returns:
      A nn.Module that maps beta vectors [B, N_vocab] to CpGPT embeddings [B, D].
      If you want an additional projection head to reach `out_features`, add it in the caller.
    """
    bb = CpGPTBackbone(
        dependencies_root=str(checkpoint_root),
        model_name=kwargs.get("model_name", "cancer"),
        species_dir=kwargs.get("species_dir", "human"),
        species_key=kwargs.get("species_key", "homo_sapiens"),
        dna_llm=kwargs.get("dna_llm", "nucleotide-transformer-v2-500m-multi-species"),
        dna_context_len=int(kwargs.get("dna_context_len", 2001)),
        finetune_lora=kwargs.get("finetune", False),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

    # If caller wants a projection to out_features, wrap with a small head
    if out_features > 0 and out_features != bb.embed_dim:
        head = nn.Sequential(
            nn.Linear(bb.embed_dim, max(out_features, bb.embed_dim)),
            nn.GELU(),
            nn.Linear(max(out_features, bb.embed_dim), out_features),
        )
        return nn.Sequential(OrderedForward(bb), head)

    return bb


class OrderedForward(nn.Module):
    """
    Tiny adapter so nn.Sequential(bb, head) calls bb.forward(x) instead of expecting dicts.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
