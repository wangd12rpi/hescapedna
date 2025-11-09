# takes all different image encoders in a single class function
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Literal, List, Set

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


def _unique_linear_leaf_names(module: nn.Module) -> List[str]:
    """
    Collect unique leaf names of nn.Linear modules, to be used as PEFT LoRA targets
    with 'endswith' matching. This is robust across typical ViT/MLP naming schemes.
    """
    names: Set[str] = set()
    for name, m in module.named_modules():
        leaf = name.split(".")[-1]
        if leaf:
            names.add(leaf)
    targets = sorted(names)
    print("Unique linear leaf names:", targets)

    targets = ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
    return targets


class ImageEncoder(nn.Module):
    """ImageEncoder that wraps GigaPath tile/slide encoders and adds a projection head."""

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

        # Finetuning switches (explicit args take precedence over 'finetune')
        self.finetune_tile: bool = kwargs.get("finetune_tile", finetune)
        self.finetune_slide: bool = kwargs.get("finetune_slide", finetune)

        # Enable cache only when the tile encoder is frozen
        cache_tiles = not self.finetune_tile

        # Build pretrained trunks
        tile_encoder, slide_encoder, self.total_blocks = self._build_trunks(self.model_name)

        # Freeze when not finetuning
        if not self.finetune_tile:
            self._freeze_module(tile_encoder)
        if not self.finetune_slide:
            self._freeze_module(slide_encoder)

        # Apply finetuning adapters
        tile_encoder, slide_encoder = self.get_ft_model(
            model_name,
            tile_encoder,
            slide_encoder,
            finetune_tile=self.finetune_tile,
            finetune_slide=self.finetune_slide,
        )

        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder
        self.trunk = _TrunkWrapper(self.tile_encoder, self.slide_encoder)
        self.trunks = (self.tile_encoder, self.slide_encoder)
        self.use_tile_cache = cache_tiles

        # Projection head for slide features
        self.proj = proj
        self.head = self._build_head(proj, 768, embed_dim)

        # Initialize cache for tile embeddings if frozen
        self.tile_cache = EmbeddingCache("tile_embeddings") if self.use_tile_cache else None

    def _build_trunks(
        self,
        model_name: str,
        **_: Any,
    ) -> tuple[nn.Module, nn.Module, int]:
        """
        Build the trunk (backbone) models for image encoding.
        Returns (tile_encoder, slide_encoder, total_blocks).
        """
        if model_name == "gigapath":
            tile_encoder, slide_encoder = _build_gigapath_model()
            print(f"Successfully loaded GigaPath tile and slide encoders for {model_name}")
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
        """
        Configure finetuning strategy:
          - tile encoder: LoRA via PEFT when finetune_tile is True
          - slide encoder: **LoRA via PEFT** when finetune_slide is True (base stays frozen)
        """
        if model_name == "gigapath":
            # Tile encoder LoRA
            if finetune_tile:
                print("**LoRA Enabled for Tile Encoder**")
                tile_targets = _unique_linear_leaf_names(tile_encoder)
                print("Tile encoder LoRA targets:", tile_targets)
                tile_encoder = get_peft_model(
                    tile_encoder,
                    LoraConfig(
                        r=16,
                        lora_alpha=16,
                        target_modules=tile_targets,
                        lora_dropout=0.1,
                        bias="none",
                    ),
                )

            # Slide encoder LoRA (base frozen; adapters trainable)
            if finetune_slide:
                print("**LoRA Enabled for Slide Encoder**")
                slide_targets = _unique_linear_leaf_names(slide_encoder)
                print("Slide encoder LoRA targets:", slide_targets)
                slide_encoder = get_peft_model(
                    slide_encoder,
                    LoraConfig(
                        r=16,
                        lora_alpha=16,
                        target_modules=slide_targets,
                        lora_dropout=0.1,
                        bias="none",
                    ),
                )
        else:
            raise ValueError(f"Unknown model name for finetuning: {model_name}")

        return tile_encoder, slide_encoder

    def _build_head(self, proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        """Build a projection head (Linear, MLP, Transformer, or MoE)."""
        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["linear"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        elif proj == "transformer":
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
        else:
            raise ValueError(f"Unknown projection type: {proj}")

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
        """
        Load tile embeddings from cache or compute if not cached.
        Uses no_grad and cache only when the tile encoder is frozen; when LoRA is active,
        we must compute with autograd and without caching.
        """

        def compute_tile_embeddings():
            image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith(".png")]
            if self.finetune_tile:
                # Finetuning path: preserve autograd graph and the caller's train/eval mode.
                return pipeline.run_inference_with_tile_encoder(image_paths, tile_encoder, batch_size=64)
            else:
                # Frozen path: faster, deterministic inference
                prev_training = tile_encoder.training
                try:
                    tile_encoder.eval()
                    with torch.no_grad():
                        return pipeline.run_inference_with_tile_encoder(image_paths, tile_encoder, batch_size=64)
                finally:
                    tile_encoder.train(prev_training)

        if self.tile_cache is None:
            return compute_tile_embeddings()
        return self.tile_cache.get_or_compute(slide_dir, compute_tile_embeddings)

    def forward(self, x):
        embeds = []

        if self.proj in ["mlp", "linear", "moe", "transformer"]:
            for slide_dir in x:
                # 1) Tile embeddings (with or without cache/autograd)
                tile_encoder_outputs = self._load_tile_embeddings_cached(slide_dir, self.tile_encoder)

                # 2) Slide-level embedding (preserve autograd; pipeline keeps caller's train/eval mode)
                slide_out = pipeline.run_inference_with_slide_encoder(
                    tile_encoder_outputs["tile_embeds"],
                    tile_encoder_outputs["coords"],
                    self.slide_encoder,
                )

                slide_embed = slide_out["last_layer_embed"]
                embeds.append(slide_embed)

            # Stack to [B, D]
            embeds = torch.stack(
                [e.squeeze(0) if isinstance(e, torch.Tensor) and e.dim() > 1 else e for e in embeds], dim=0
            )

            # Projection head
            embeds = self.head(embeds)
        else:
            print(f"[ENCODER DEBUG] No branch matched for proj={self.proj}, returning raw x={type(x)}")

        return embeds


if __name__ == "__main__":
    # Minimal smoke test (not executed in training)
    encoder = ImageEncoder(
        model_name="gigapath",
        finetune=True,
        embed_dim=128,
        proj="mlp",
        finetune_tile=True,
        finetune_slide=True,  # LoRA on slide encoder as well
    )
    print_trainable_parameters("gigapath", encoder)
