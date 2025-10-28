# takes all different image encoders in a single class function
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Literal

import torch
import torch.nn as nn
from gigapath import pipeline
from peft import LoraConfig, get_peft_model
from timm.layers import Mlp

from hescape.models._cache import EmbeddingCache
from hescape.models._utils import print_trainable_parameters
from hescape.models.image_models._gigapath import _build_gigapath_model
from hescape.models.image_models._utils import freeze_batch_norm_2d
from .moe import MoE



@contextmanager
def quiet_encoding():
    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            yield
        finally:
            logging.disable(old_disable)


class _JointBackbone(nn.Module):
    """Wrapper to expose tile and slide encoders under the historic attribute names."""

    def __init__(self, tile_encoder: nn.Module, slide_encoder: nn.Module):
        super().__init__()
        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder


class _TrunkWrapper(nn.Module):
    """Compatibility wrapper to preserve checkpoint key structure."""

    def __init__(self, tile_encoder: nn.Module, slide_encoder: nn.Module):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = _JointBackbone(tile_encoder, slide_encoder)

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


class ImageEncoder(nn.Module):
    """ImageEncoder that wraps timm models."""

    def __init__(
        self,
        model_name: Literal["gigapath"] | str,
        finetune: bool = False,
        embed_dim: int = -1,
        proj: str = "mlp",
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name

        tile_finetune = kwargs.get("finetune_tile", finetune)
        slide_finetune = kwargs.get("finetune_slide", finetune)
        cache_tiles = not tile_finetune

        tile_encoder, slide_encoder, self.total_blocks = self._build_trunks(
            self.model_name,
        )

        if not tile_finetune:
            self._freeze_module(tile_encoder)
        if not slide_finetune:
            self._freeze_module(slide_encoder)

        tile_encoder, slide_encoder = self.get_ft_model(
            model_name,
            tile_encoder,
            slide_encoder,
            finetune_tile=tile_finetune,
            finetune_slide=slide_finetune,
        )

        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder
        self.trunk = _TrunkWrapper(self.tile_encoder, self.slide_encoder)
        self.trunks = (self.tile_encoder, self.slide_encoder)
        self.finetune_tile = tile_finetune
        self.finetune_slide = slide_finetune
        self.use_tile_cache = cache_tiles
        # Build projection head
        self.proj = proj
        self.head = self._build_head(proj, 768, embed_dim)

        # Initialize cache for tile embeddings
        self.tile_cache = EmbeddingCache('tile_embeddings') if self.use_tile_cache else None

        # return hook

    def _build_trunks(
        self,
        model_name: str,
        **_: Any,
    ) -> tuple[nn.Module, nn.Module, int]:
        """
        Build the trunk (backbone) model for image encoding.
        Returns (trunk_module, total_blocks).
        """

        if model_name == "gigapath":
            tile_encoder, slide_encoder = _build_gigapath_model()
            print(f"Successfully loaded GigaPath slide encoder for {model_name}")
            total_blocks = 12

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return tile_encoder, slide_encoder, total_blocks

    def get_ft_model(
        self,
        model_name: str,
        tile_encoder: nn.Module,
        slide_encoder: nn.Module,
        *,
        finetune_tile: bool,
        finetune_slide: bool,
    ) -> tuple[nn.Module, nn.Module]:
        # Define LoRA configurations for each model
        lora_configs = {"gigapath": {"r": 4, "lora_alpha": 16, "target_modules": ["proj"]}}

        if finetune_slide and model_name in lora_configs:
            print("**LoRA Enabled for Slide Encoder**")
            config = lora_configs[model_name]
            slide_encoder = get_peft_model(
                slide_encoder,
                LoraConfig(
                    r=config["r"],
                    lora_alpha=config["lora_alpha"],
                    target_modules=config["target_modules"],
                    lora_dropout=0.1,
                    bias="none",
                ),
            )

        return tile_encoder, slide_encoder

    def _build_head(self, proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        """Build a projection head (Linear or MLP)."""
        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["linear"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        elif proj == "transformer":
            # Add transformer specific layers here (From torch maybe)
            encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8, dim_feedforward=embed_dim)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            head_layers["transformer"] = transformer_encoder
            head_layers["linear"] = nn.Linear(in_features, embed_dim)
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

        return nn.Sequential(head_layers)

    def _freeze_module(self, module: nn.Module, freeze_bn_stats: bool = True) -> None:
        for param in module.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(module)

    def freeze(self, freeze_bn_stats: bool = True):
        """Freeze model params."""
        self._freeze_module(self.tile_encoder, freeze_bn_stats=freeze_bn_stats)
        self._freeze_module(self.slide_encoder, freeze_bn_stats=freeze_bn_stats)

    def _load_tile_embeddings_cached(self, slide_dir: str, tile_encoder) -> dict:
        """Load tile embeddings from cache or compute if not cached."""
        def compute_tile_embeddings():
            image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]
            # with quiet_encoding():
            with torch.no_grad():
                return pipeline.run_inference_with_tile_encoder(
                    image_paths, tile_encoder, batch_size=64
                )

        if self.tile_cache is None:
            return compute_tile_embeddings()
        return self.tile_cache.get_or_compute(slide_dir, compute_tile_embeddings)

    def forward(self, x):
        embeds = []

        if self.proj in ["mlp", "linear", "moe"]:
            for slide_dir in x:
                # Load tile embeddings from cache or compute
                tile_encoder_outputs = self._load_tile_embeddings_cached(slide_dir, self.tile_encoder)

                slide_embeds = pipeline.run_inference_with_slide_encoder(
                    slide_encoder_model=self.slide_encoder,
                    **tile_encoder_outputs
                )

                embeds.append(slide_embeds['last_layer_embed'])

            # Stack to preserve gradient flow through batch dimension
            embeds = torch.stack([e.squeeze(0) if e.dim() > 1 else e for e in embeds], dim=0)
            # print(embeds.shape)
            embeds = self.head(embeds)

        else:
            print(f"[ENCODER DEBUG] No branch matched for proj={self.proj}, returning raw x={x.shape}")

        return embeds


if __name__ == "__main__":
    # Create an instance of the ImageEncoder class

    for model_name in ["optimus"]:  # , "uni", "ctranspath", "optimus", "conch", "gigapath"]:
        encoder = ImageEncoder(
            model_name=model_name,
            finetune=True,
            embed_dim=128,
            proj="mlp",
            checkpoint_path="/p/project1/hai_spatial_clip/pretrain_weights/image",
        )

        # encoder.freeze()

        encoder = encoder.to("cuda")
        dummy_input = torch.Tensor(1, 3, 224, 224).uniform_().to("cuda")
        output = encoder(dummy_input)
        print(output.shape)  # Output shape: [batch_size, num_features]

        # print parameter names which have gradient set to True
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                print(name)

        print_trainable_parameters(model_name, encoder)
