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
import logging, os

from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def quiet_encoding():
    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            yield
        finally:
            logging.disable(old_disable)


class DnaMethEncoder(nn.Module):
    """
    Minimal DNA-methylation encoder for hescape that wraps your CpGPT runner.

    Forward expects a list of file paths to beta TSV files:
      - Each file has two columns: CpG_Site  Beta_Value  (tab-separated)

    The trunk (CpGPT) is always frozen. You can pick a projection head:
      proj = "identity" | "linear" | "mlp"
    """

    def __init__(
        self,
        input_sites: int | None,                 # kept for API parity; CpGPT uses its own vocab
        model_name: str = "cancer",
        embed_dim: int = 128,                    # projection output dimension
        finetune: bool = False,                  # ignored; CpGPT stays frozen
        proj: str = "identity",
        checkpoint_path: str | None = None,      # root folder holding cpgpt_files/dependencies
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

        # Build trunk (CpGPT runner). No finetuning.
        # out_features is unused here; head is separate in this class.
        with quiet_encoding():
            self.trunk = _build_cpgpt_model(
                checkpoint_root=checkpoint_root,
                in_features=input_sites or 0,
                out_features=embed_dim,
                model_name=model_name,
                cache_embeddings=kwargs.get("cache_embeddings", not finetune),
            )
        self.cache_embeddings = bool(getattr(self.trunk, "cache_embeddings", False))

        # CpGPT embeddings dimension resolved from config when available
        self.trunk_dim = getattr(self.trunk, "embedding_dim", 128)

        # Projection head
        self.head = self._build_head(self.proj, self.trunk_dim, self.embed_dim).to(self.device)

        # Freeze trunk
        self.freeze()

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

    def freeze(self):
        """Trunk (CpGPT) stays frozen; only the head can be trained if desired."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, beta_paths: List[str]) -> torch.Tensor:
        """
        beta_paths: list of local file paths to TSV beta files.
        Returns:
          features [B, embed_dim]
        """
        # CpGPT embeddings [B, 128] - caching is now handled inside CpGPTRunner
        # with quiet_encoding():
        emb = self.trunk.encode_beta_files(beta_paths).to(self.device)
        # Project if required
        out = self.head(emb)
        return out

    # Optional utility for debugging
    def print_trainable(self, tag: str = "dnameth"):
        print_trainable_parameters(tag, self)


# quick manual test
if __name__ == "__main__":
    # Example:
    #   checkpoint_path="/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files"
    #   beta_paths = ["/path/sample1.txt", "/path/sample2.txt", ...]
    encoder = DnaMethEncoder(
        input_sites=None,
        model_name="cancer",
        embed_dim=128,
        finetune=False,
        proj="identity",
        checkpoint_path="/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files",
    )
    # embeddings = encoder(beta_paths)  # [B, 128]
    # print(embeddings.shape)
