# hescape/models/dnameth_models/dnameth_encoder.py
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from hescape.models._utils import print_trainable_parameters
from hescape.models.dnameth_models._cpgpt import _build_cpgpt_model


class DnaMethEncoder(nn.Module):
    """
    DNA-methylation encoder wrapper that integrates CpGPT.

    Forward expects a list of file paths to beta TSV files:
      - Each file has two columns: CpG_Site  Beta_Value  (tab-separated)

    Finetuning policy:
      - If finetune=False (default): CpGPT trunk is frozen and embeddings are produced via a cached, fast path.
      - If finetune=True: Inject LoRA adapters into CpGPT; forward runs **in-graph** so LoRA weights receive gradients.
        The CpGPT base stays frozen; only LoRA params (and the projection head) are trainable.
    """

    def __init__(
        self,
        input_sites: int | None,
        model_name: str = "cancer",
        embed_dim: int = 128,
        finetune: bool = False,
        proj: str = "identity",
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must point to your cpgpt_files root")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_root = Path(checkpoint_path)
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.proj = proj
        self.finetune = bool(finetune)

        # Build trunk (CpGPT runner). Use LoRA adapters only if finetune=True.
        self.trunk = _build_cpgpt_model(
            checkpoint_root=checkpoint_root,
            in_features=input_sites or 0,
            out_features=embed_dim,
            model_name=model_name,
            cache_embeddings=kwargs.get("cache_embeddings", not self.finetune),
            enable_lora=self.finetune,
            lora_r=8
        )

        # Register the underlying CpGPT Lightning module as a nn.Module child
        # so its LoRA parameters are visible to the optimizer.
        cpgpt_model = getattr(self.trunk, "model", None)
        if not isinstance(cpgpt_model, nn.Module):
            raise RuntimeError("CpGPT runner does not expose a valid nn.Module under 'model'.")
        self.cpgpt: nn.Module = cpgpt_model  # registration via attribute assignment

        # Resolve CpGPT embedding size when available
        self.trunk_dim = int(getattr(self.trunk, "embedding_dim", 128))

        # Projection head (trainable)
        self.head = self._build_head(self.proj, self.trunk_dim, self.embed_dim).to(self.device)

        # Configure trainability (freeze/unfreeze policy)
        self._configure_trainability()

    @staticmethod
    def _build_head(proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        head_layers = OrderedDict()
        if proj == "identity":
            head_layers["proj"] = nn.Identity()
        elif proj == "linear":
            head_layers["proj"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        else:
            raise ValueError(f"Unknown projection type: {proj}")
        return nn.Sequential(head_layers)

    def _configure_trainability(self) -> None:
        """
        - finetune=False: freeze entire CpGPT and train only the projection head.
        - finetune=True: keep CpGPT base frozen, unfreeze only LoRA params + projection head.
        """
        # First freeze everything to a known state
        for p in self.parameters():
            p.requires_grad = False

        # Projection head always trainable
        for p in self.head.parameters():
            p.requires_grad = True

        if self.finetune:
            # Enable only LoRA parameters inside the CpGPT module
            for name, p in self.cpgpt.named_parameters():
                if "lora_" in name.lower():
                    p.requires_grad = True

        # Optional: print a quick summary
        print_trainable_parameters("dnameth_encoder", self)

    def forward(self, beta_paths: List[str]) -> torch.Tensor:
        """
        beta_paths: list of local file paths to TSV beta files.
        Returns:
          features [B, embed_dim] on self.device
        """
        if self.finetune:
            # In-graph path so LoRA adapters can learn
            emb = self.trunk.encode_beta_files_autograd(beta_paths)
        else:
            # Fast frozen path (cached)
            emb = self.trunk.encode_beta_files(beta_paths).to(self.device)

        out = self.head(emb)
        return out

    # Optional utility for debugging
    def print_trainable(self, tag: str = "dnameth"):
        print_trainable_parameters(tag, self)


# quick manual test (commented out to avoid side effects on import)
# if __name__ == "__main__":
#     encoder = DnaMethEncoder(
#         input_sites=None,
#         model_name="cancer",
#         embed_dim=128,
#         finetune=True,
#         proj="identity",
#         checkpoint_path="/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files",
#     )
#     encoder.print_trainable()
