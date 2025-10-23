# takes all different image encoders in a single class function
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from timm.layers import Mlp

from hescape.models._utils import print_trainable_parameters
from hescape.models.image_models._utils import freeze_batch_norm_2d
from hescape.models.image_models._gigapath import _build_gigapath_model
from gigapath import pipeline
import os
from .moe import MoE
from pathlib import Path
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr

project_root = next(p for p in Path(__file__).parents if (p / '.git').exists())



@contextmanager
def quiet_encoding():
    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            yield
        finally:
            logging.disable(old_disable)

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

        self.trunks, self.total_blocks = self._build_trunks(self.model_name, **kwargs)

        if not finetune:  # i.e if finetune is false, we freeze the trunk
            self.freeze()

        self.trunks = self.get_ft_model(model_name, self.trunks, lora=finetune)

        # Build projection head
        self.proj = proj
        self.head = self._build_head(proj, 768, embed_dim)

        # return hook

    def _build_trunks(self, model_name: str, **kwargs: Any) -> tuple[nn.Module, int]:
        """
        Build the trunk (backbone) model for image encoding.
        Returns (trunk_module, total_blocks).
        """

        if model_name == "gigapath":
            trunks = _build_gigapath_model()
            print(f"Successfully loaded GigaPath slide encoder for {model_name}")
            total_blocks = 12

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return trunks, total_blocks

    def get_ft_model(self, model_name: str, trunks, lora: bool = False) -> object:
        # Define LoRA configurations for each model
        lora_configs = {
            "gigapath": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
        }

        if lora:
            print("**LoRA Enabled for Slide Encoder**")
            # Get the LoRA configuration for the given model
            if model_name in lora_configs.keys():
                config = lora_configs.get(model_name)
                if config:
                    # Create a LoRA configuration object
                    lora_config = LoraConfig(
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=config["target_modules"],
                        lora_dropout=0.1,
                        bias="none",
                    )
                    # Return the fine-tuned model with LoRA
                    return get_peft_model(trunks[0], lora_config), get_peft_model(trunks[1], lora_config)
                else:
                    # Handle unknown model names
                    raise ValueError(f"Unknown model name: {model_name}")

        return trunks

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

    def freeze(self, freeze_bn_stats=True):
        """Freeze model params."""
        for x in self.trunks:
            for param in x.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(x)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: list of tile paths

        Returns:
            Embeddings of shape [B, embed_dim]
        """

        tile_encoder, slide_encoder = self.trunks

        embeds = []

        if self.proj in ["mlp", "linear", "moe"]:
            for slide_dir in x:
                image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]
                with quiet_encoding():
                    tile_encoder_outputs = pipeline.run_inference_with_tile_encoder(image_paths, tile_encoder,
                                                                                    batch_size=512)
                    slide_embeds = pipeline.run_inference_with_slide_encoder(slide_encoder_model=slide_encoder,
                                                                             **tile_encoder_outputs)
                embeds.append(slide_embeds['last_layer_embed'])

            # Stack to preserve gradient flow through batch dimension
            embeds = torch.stack([e.squeeze(0) if e.dim() > 2 else e for e in embeds], dim=0)
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
