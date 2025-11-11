from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from open_clip.loss import ClipLoss, SigLipLoss
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from hescape.constants import DatasetEnum
from hescape.models._utils import print_trainable_parameters
from hescape.models.dnameth_models import DnaMethEncoder
from hescape.models.image_models import ImageEncoder



LOCAL_RANK = "LOCAL_RANK"

REGRESSION_METRICS = [
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
]

CLASSIFICATION_METRICS = [
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
]


class CLIPModel(nn.Module):
    """CLIP model."""

    def __init__(
        self,
        input_sites: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath"],
        dnameth_enc_name: str,
        loss: Literal["CLIP", "SIGLIP"],
        tile_finetune: bool,
        slide_finetune: bool,
        dnameth_finetune: bool,
        n_tissue: int | None = None,
        n_region: int | None = None,
        image_size: int = 224,
        temperature: float = 0.07,
        world_size: int = 1,
        rank: int = 0,
        cfg: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        image_encoder_kwargs: dict[str, Any] = {
            "model_name": img_enc_name,
            "embed_dim": embed_dim,
            "finetune_tile": tile_finetune,
            "finetune_slide": slide_finetune,
            "proj": kwargs.get("img_proj", "linear"),
        }
        if "img_enc_path" in kwargs:
            image_encoder_kwargs["checkpoint_path"] = kwargs["img_enc_path"]
        for optional_key in ("global_pool", "use_slide_encoder"):
            if optional_key in kwargs:
                image_encoder_kwargs[optional_key] = kwargs[optional_key]
        self.image_encoder = ImageEncoder(**image_encoder_kwargs)

        self.dnameth_enc_name = dnameth_enc_name
        self.dnameth_encoder = DnaMethEncoder(
            input_sites=input_sites,
            model_name=dnameth_enc_name,
            checkpoint_path=kwargs.get("dnameth_enc_path", kwargs.get("gene_enc_path", None)),
            finetune=dnameth_finetune,
            embed_dim=embed_dim,
            proj=kwargs.get("dnameth_proj", kwargs.get("gene_proj", "identity")),
        )

        print_trainable_parameters(img_enc_name, self.image_encoder)
        print_trainable_parameters(dnameth_enc_name, self.dnameth_encoder)

        # ------------------------
        # 2) CLIP Loss Setup
        # ------------------------
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        if loss == "CLIP":
            self.logit_bias = None
            loss_fn = ClipLoss(world_size=world_size, rank=rank)
        elif loss == "SIGLIP":
            self.logit_bias = nn.Parameter(torch.ones([]) * -10)
            loss_fn = SigLipLoss(world_size=world_size, rank=rank)
        else:
            raise ValueError(f"Loss {loss} not supported.")
        self.loss = loss_fn

    def forward(self, batch: dict[str, Tensor], norm: bool = True):
        """Forward pass: returns (img_embed, dnameth_embed, logit_scale.exp())."""
        dnameth_input = batch[DatasetEnum.DNAMETH]

        if isinstance(dnameth_input, torch.Tensor):
            raise TypeError(
                "DnaMethEncoder expects a list of beta file paths. "
                "Ensure the datamodule is configured with beta_representation='path'."
            )

        dnameth_embed = self.dnameth_encoder(dnameth_input)
        # Encode images
        img_embed = self.image_encoder(batch[DatasetEnum.IMG])

        if norm:
            return (
                F.normalize(img_embed, p=2, dim=-1),
                F.normalize(dnameth_embed, p=2, dim=-1),
                self.logit_scale.exp(),
            )

        return img_embed, dnameth_embed, self.logit_scale.exp()

    def compute_loss(self, img_embed, dnameth_embed):  # default 1.0
        """
        Compute the total loss comprising:
          - Contrastive CLIP loss.
        """
        if self.logit_bias is not None:
            contrastive_loss = self.loss(
                img_embed, dnameth_embed, logit_scale=self.logit_scale.exp(), logit_bias=self.logit_bias
            )
        else:
            # print(img_embed.shape, dnameth_embed.shape)
            contrastive_loss = self.loss(img_embed, dnameth_embed, logit_scale=self.logit_scale.exp())

        return contrastive_loss
