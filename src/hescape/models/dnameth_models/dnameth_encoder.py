# hescape/models/dnameth_encoder.py
from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from hescape.models.gexp_models.moe import MoE
from hescape.models.gexp_models._utils import freeze_batch_norm_2d
from hescape.models._utils import print_trainable_parameters
from hescape.models.dnameth_models import _build_cpgpt_model


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


class MoEHeadAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        num_experts: int = 4,
        k: int = 2,
        head_size: int | None = None,
        cvloss: float = 0.01,
        switchloss: float = 0.01,
        zloss: float = 1e-4,
        noisy_gating: bool = True,
        acc_aux_loss: bool = False,
    ):
        super().__init__()
        head_size = head_size or (2 * in_features)
        self.moe = MoE(
            input_size=in_features,
            head_size=head_size,
            num_experts=num_experts,
            k=k,
            cvloss=cvloss,
            switchloss=switchloss,
            zloss=zloss,
            noisy_gating=noisy_gating,
            acc_aux_loss=acc_aux_loss,
        )
        self.proj = nn.Linear(in_features, embed_dim)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y, aux = self.moe(x)
        self.last_aux_loss = aux
        y = y.squeeze(1)
        return self.proj(y)


class DnaMethEncoder(nn.Module):
    """
    DNA methylation encoder that wraps CpGPT as trunk and adds a projection head.

    Inputs
    ------
    input_sites: number of CpG sites. The input x must be [B, input_sites] in the same order used across the dataset.
    model_name:  'cpgpt' currently (reserved for future variants).
    embed_dim:   target embedding size after the head.
    finetune:    if False, CpGPT weights are frozen. If True, you can inject LoRA outside this class.
    proj:        'identity' | 'linear' | 'mlp' | 'moe'
    checkpoint_path: directory or file to a CpGPT .ckpt

    Optional context features
    -------------------------
    n_region, n_tissue: add one‑hot covariates to the trunk output before the head.
    """

    def __init__(
        self,
        input_sites: int,
        model_name: str = "cpgpt",
        embed_dim: int = 128,
        finetune: bool = False,
        proj: str = "identity",
        n_region: int | None = None,
        n_tissue: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_sites = int(input_sites)
        self.model_name = model_name
        self.embed_dim = int(embed_dim)
        self.n_region = n_region
        self.n_tissue = n_tissue
        self.proj_kind = proj

        checkpoint_root = Path(kwargs.get("checkpoint_path", ""))  # required
        if not checkpoint_root.exists():
            raise FileNotFoundError(
                f"checkpoint_path='{checkpoint_root}' does not exist. "
                f"Provide a CpGPT .ckpt file or a folder containing it."
            )

        # trunk
        # kwargs pass-through: learned_site_emb, seq_dim, freeze_cpgpt, etc.
        trunk = _build_cpgpt_model(
            checkpoint_root=checkpoint_root,
            in_features=self.input_sites,
            out_features=self.embed_dim,  # trunk will project to this before head
            **kwargs,
        )
        self.trunk = trunk

        # report features from trunk
        num_features = self.embed_dim

        # freeze or leave trainable
        if not finetune:
            self.freeze()
        else:
            self.unfreeze_trunk()  # keep as is; LoRA injection can be applied by caller

        # one‑hot covariates
        if self.n_region is not None:
            self.region_onehot = partial(one_hot, n_cat=self.n_region)
            num_features += self.n_region
        else:
            self.region_onehot = None

        if self.n_tissue is not None:
            self.tissue_onehot = partial(one_hot, n_cat=self.n_tissue)
            num_features += self.n_tissue
        else:
            self.tissue_onehot = None

        # head
        self.head = self._build_head(self.proj_kind, num_features, self.embed_dim)

    def _build_head(self, proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["proj"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        elif proj == "identity":
            head_layers["proj"] = nn.Identity()
        elif proj == "moe":
            head_layers["moe"] = MoEHeadAdapter(
                in_features=in_features,
                embed_dim=embed_dim,
                num_experts=4,
                k=2,
                head_size=2 * in_features,
                cvloss=0.01,
                switchloss=0.01,
                zloss=1e-4,
                noisy_gating=True,
                acc_aux_loss=False,
            )
        else:
            raise ValueError(f"Unknown projection type: {proj}")
        return nn.Sequential(head_layers)

    def freeze(self, freeze_bn_stats: bool = True):
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.trunk.eval()
        if freeze_bn_stats:
            # CpGPT does not have 2D batchnorm, but this is harmless
            freeze_batch_norm_2d(self.trunk)

    def unfreeze_trunk(self):
        for p in self.trunk.parameters():
            p.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,  # [B, input_sites] beta values with NaNs allowed
        region: torch.Tensor | None = None,
        tissue: torch.Tensor | None = None,
        seq_emb: torch.Tensor | None = None,  # optional [B, N, E] if using real DNA embeddings
        chroms: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # trunk embedding
        z = self.trunk(
            x,
            seq_emb=seq_emb,
            chroms=chroms,
            positions=positions,
            out_dim=self.embed_dim,
        )  # [B, embed_dim]

        # optional covariates
        feats = [z]
        if self.region_onehot is not None and region is not None:
            feats.append(self.region_onehot(region[:, None]))
        if self.tissue_onehot is not None and tissue is not None:
            feats.append(self.tissue_onehot(tissue[:, None]))

        out = torch.cat(feats, dim=1) if len(feats) > 1 else z
        out = self.head(out)
        return out


if __name__ == "__main__":
    # quick smoke test (uses learned site embeddings, no CpGPT ckpt is actually loaded here)
    # Replace 'checkpoint_path' with your CpGPT .ckpt or folder.
    try:
        enc = DnaMethEncoder(
            input_sites=5000,
            model_name="cpgpt",
            embed_dim=128,
            finetune=False,
            proj="mlp",
            checkpoint_path="/path/to/cpgpt_ckpt_dir_or_file",
            learned_site_emb=True,
            seq_dim=256,
        ).cuda()

        x = torch.rand(4, 5000, device="cuda")
        out = enc(x)
        print(out.shape)
        print_trainable_parameters("cpgpt", enc)
    except Exception as e:
        print("Smoke test skipped:", e)
