# hescape/models/dnameth_models/_cpgpt.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_ckpt_file(root: Path, candidates: tuple[str, ...] = ("cpgpt.ckpt", "last.ckpt", "model.ckpt")) -> Optional[Path]:
    if root.is_file():
        return root
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    # fallback: first .ckpt under root
    ckpts = list(root.glob("*.ckpt"))
    return ckpts[0] if ckpts else None


class CpGPTBackbone(nn.Module):
    """
    Minimal backbone that uses CpGPT to turn a beta vector [B, N] into a sample embedding [B, D].

    Two modes:
      - learned_site_emb=True: create a learnable site embedding table [N, seq_dim] to fill
        CpGPT's `sequence_embeddings`. This works without genomic coordinates.
      - learned_site_emb=False: expect caller to pass real `sequence_embeddings` via forward(..., seq_emb=...).

    Notes:
      - CpGPT expects fields: meth, sequence_embeddings, chroms, positions, mask_na.
        We provide zeros for chroms/positions when using learned site embeddings.
      - The returned embedding dimension is inferred at runtime on the first forward.
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        num_sites: int,
        out_dim: int = 128,
        learned_site_emb: bool = True,
        seq_dim: int = 256,
        freeze_cpgpt: bool = True,
    ):
        super().__init__()
        ckpt_path = Path(ckpt_path)
        try:
            # CpGPT is a Lightning module with .net inside
            from cpgpt.models.module import CpGPTLitModule  # type: ignore
        except Exception as e:
            raise ImportError(
                "CpGPT not found. Install CpGPT and ensure it is on PYTHONPATH. "
                "Repo: https://github.com/lcamillo/CpGPT"
            ) from e

        # load lightning checkpoint
        if ckpt_path.is_dir():
            resolved = _find_ckpt_file(ckpt_path)
            if resolved is None:
                raise FileNotFoundError(f"No CpGPT .ckpt found in {ckpt_path}")
            ckpt_path = resolved

        self.lit: Any = CpGPTLitModule.load_from_checkpoint(checkpoint_path=str(ckpt_path), strict=False, map_location="cpu")
        self.net: nn.Module = self.lit.net  # the actual PyTorch net with encode_sample(...)
        self.num_sites = int(num_sites)

        # site embeddings to fake CpGPT's dna sequence embeddings if needed
        self.learned_site_emb = learned_site_emb
        if learned_site_emb:
            self.site_table = nn.Embedding(num_embeddings=self.num_sites, embedding_dim=seq_dim)
            nn.init.normal_(self.site_table.weight, mean=0.0, std=0.02)
            self.seq_dim = seq_dim
        else:
            self.site_table = None
            self.seq_dim = None  # must be provided at forward time via seq_emb

        # linear projection to the requested out_dim, will be set after probing once
        self._sample_dim: Optional[int] = None
        self.proj = None  # will initialize on first call

        if freeze_cpgpt:
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()  # eval switches off dropout but does NOT block grads if you unfreeze later

    @property
    def sample_dim(self) -> int:
        if self._sample_dim is None:
            raise RuntimeError("Call forward once to infer sample embedding dimension, or set it manually.")
        return self._sample_dim

    def _ensure_proj(self, sample_dim: int, out_dim: int):
        if self.proj is None:
            self.proj = nn.Linear(sample_dim, out_dim)
        else:
            # if already created for a different dim, reinit
            if self.proj.in_features != sample_dim or self.proj.out_features != out_dim:
                self.proj = nn.Linear(sample_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,                  # [B, N] beta values with possible NaNs
        seq_emb: torch.Tensor | None = None,   # [B, N, E] real DNA embeddings if available
        chroms: torch.Tensor | None = None,    # ignored when learned_site_emb=True
        positions: torch.Tensor | None = None, # ignored when learned_site_emb=True
        out_dim: int | None = None,
        train_mode: bool | None = None,
    ) -> torch.Tensor:
        """
        Returns [B, out_dim] embedding.
        """
        B, N = x.shape
        device = x.device
        # choose CpGPT train/eval only if caller explicitly asks
        if train_mode is not None:
            self.net.train(mode=train_mode)

        # build mask for NaNs and zero-fill
        mask_na = torch.isnan(x)
        meth = torch.nan_to_num(x, nan=0.0)

        # build sequence embeddings
        if self.learned_site_emb:
            # learned table by site index [0..N-1], broadcast to batch
            idx = torch.arange(N, device=device, dtype=torch.long)
            base = self.site_table(idx)                  # [N, E]
            seq_embeddings = base.unsqueeze(0).expand(B, N, -1)  # [B, N, E]
            chroms = torch.zeros(B, N, dtype=torch.int32, device=device)
            positions = torch.zeros(B, N, dtype=torch.int32, device=device)
        else:
            if seq_emb is None:
                raise ValueError("seq_emb must be provided when learned_site_emb is False.")
            seq_embeddings = seq_emb  # [B, N, E]
            if chroms is None or positions is None:
                # create placeholders if not provided
                chroms = torch.zeros(B, N, dtype=torch.int32, device=device)
                positions = torch.zeros(B, N, dtype=torch.int32, device=device)

        # CpGPT expects boolean mask of same shape
        input_data = {
            "meth": meth,                                   # float [B,N]
            "sequence_embeddings": seq_embeddings,          # float [B,N,E]
            "chroms": chroms,                               # int   [B,N]
            "positions": positions,                         # int   [B,N]
            "mask_na": mask_na,                             # bool  [B,N]
        }

        # sample embedding
        sample_embed: torch.Tensor = self.net.encode_sample(**input_data)  # [B, D*] unknown yet
        Dstar = sample_embed.shape[-1]

        # lazily build projection if needed
        target_dim = out_dim if out_dim is not None else Dstar
        self._ensure_proj(Dstar, target_dim)
        self._sample_dim = Dstar

        # normalize then project to requested dimension
        sample_embed = F.layer_norm(sample_embed, normalized_shape=(Dstar,))
        out = self.proj(sample_embed) if self.proj is not None else sample_embed
        return out


def _build_cpgpt_model(
    checkpoint_root: Path | str,
    in_features: int,
    out_features: int,
    learned_site_emb: bool = True,
    seq_dim: int = 256,
    freeze_cpgpt: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory used by your encoder wrapper.

    Args
    ----
    checkpoint_root: path to a CpGPT lightning .ckpt file or a directory containing it.
    in_features:     number of CpG sites (must match your vector length).
    out_features:    target trunk output size before the projection head in the outer encoder.

    Returns
    -------
    nn.Module that maps [B, in_features] -> [B, out_features]
    """
    ckpt_path = Path(checkpoint_root)
    if ckpt_path.is_dir():
        found = _find_ckpt_file(ckpt_path)
        if found is None:
            raise FileNotFoundError(f"No CpGPT .ckpt found under {ckpt_path}")
        ckpt_path = found
    trunk = CpGPTBackbone(
        ckpt_path=ckpt_path,
        num_sites=in_features,
        out_dim=out_features,
        learned_site_emb=learned_site_emb,
        seq_dim=seq_dim,
        freeze_cpgpt=freeze_cpgpt,
    )
    return trunk
