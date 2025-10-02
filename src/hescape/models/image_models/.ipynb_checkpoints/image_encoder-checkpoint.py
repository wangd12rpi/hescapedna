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
from hescape.models.image_models._conch import _build_conch_model
from hescape.models.image_models._ctranspath import _build_ctranspath_model
from hescape.models.image_models._h0_mini import _build_h0_mini_model
from hescape.models.image_models._utils import freeze_batch_norm_2d

from .moe import MoE

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
        model_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath", "h0-mini"] | str,
        finetune: bool = False,
        embed_dim: int = -1,
        proj: str = "mlp",
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name

        # Build trunk model
        checkpoint_root = Path(kwargs.get("checkpoint_path", ""))
        self.trunk, self.total_blocks = self._build_trunk(self.model_name, checkpoint_root, **kwargs)

        if not finetune:  # i.e if finetune is false, we freeze the trunk
            self.freeze()

        self.trunk = self.get_ft_model(model_name, self.trunk, lora=finetune)

        # Build projection head
        self.proj = proj
        self.head = self._build_head(proj, self.trunk.num_features, embed_dim)

        # return hook

    def _build_trunk(self, model_name: str, checkpoint_root: Path, **kwargs: Any) -> tuple[nn.Module, int]:
        """
        Build the trunk (backbone) model for image encoding.
        Returns (trunk_module, total_blocks).
        """
        if model_name == "densenet":
            trunk = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
            print(f"Successfully loaded weights for {model_name}")
            total_blocks = 4  # Fine-tune up to 2

        elif model_name == "ctranspath":
            trunk = _build_ctranspath_model()
            checkpoint_path = checkpoint_root / model_name / "ctranspath.pth"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 4  # Fine-tune up to 2

        elif model_name == "uni":
            trunk = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 24  # Fine-tune up to 8

        elif model_name == "optimus":
            trunk = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False
            )
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 40  # Fine-tune up to 12

        elif model_name == "h0-mini":  # to be refined
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            model = _build_h0_mini_model(str(checkpoint_path))
            print(f"Successfully loaded weights for {model_name}")

            trunk = model.trunk

            total_blocks = 12  # Fine-tune up to 12

        elif model_name == "conch":
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            model = _build_conch_model(str(checkpoint_path))
            print(f"Successfully loaded weights for {model_name}")

            trunk = model.visual.trunk

            total_blocks = 12

        elif model_name == "gigapath":
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            # total_blocks may differ, set it according to your needs
            total_blocks = 12  # Example

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return trunk, total_blocks

    def get_ft_model(self, model_name: str, trunk, lora: bool = False) -> object:
        """
        Returns a fine-tuned model based on the given model name and trunk.

        Args:
            model_name (str): The name of the model.
            trunk: The trunk model.
            lora (bool, optional): Whether to use LoRA. Defaults to False.

        Returns
        -------
            object: The fine-tuned model.
        """
        # Define LoRA configurations for each model
        lora_configs = {
            "ctranspath": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "attn.proj"]},
            "uni": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "conch": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "optimus": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "h0-mini": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "gigapath": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
        }

        if lora:
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
                    return get_peft_model(trunk, lora_config)
                else:
                    # Handle unknown model names
                    raise ValueError(f"Unknown model name: {model_name}")

        # If LoRA is not enabled, return the original trunk
        return trunk  # it simply returns the trunk for densenet

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
            state_dict_path = "/lus/lfs1aip1/home/u5t/chuhansong.u5t/results/human_lung_healthy_panel/local/checkpoints/last.ckpt"
            
            # Hydra/OmegaConf fix: 添加 safe globals 以避免 UnpicklingError
            import torch
            from omegaconf import DictConfig
            torch.serialization.add_safe_globals([DictConfig])
            
            full_ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=False)
            
            # 提取 model state_dict（PyTorch Lightning/Hydra ckpt 通常是 {'state_dict': state_dict, 'epoch': ..., ...}）
            if isinstance(full_ckpt, dict) and 'state_dict' in full_ckpt:
                full_state_dict = full_ckpt['state_dict']
                print(f"[DEBUG] Extracted 'state_dict' from Lightning ckpt, keys: {len(full_state_dict)}")
            elif isinstance(full_ckpt, dict) and 'model' in full_ckpt:
                full_state_dict = full_ckpt['model']
                print(f"[DEBUG] Extracted 'model' from Hydra ckpt, keys: {len(full_state_dict)}")
            else:
                full_state_dict = full_ckpt
                print(f"[DEBUG] Direct ckpt load, keys: {len(full_state_dict)}")
            
            # 打印 ckpt keys 样本以调试前缀
            print(f"[DEBUG] Sample ckpt keys: {list(full_state_dict.keys())[:5]}")
            
            # 过滤 MoE 相关的 key（捕获所有 head.moe. 子树，包括 moe. 和 proj.）
            moe_prefix = "model.image_encoder.head.moe."
            moe_subdict = {}
            for k, v in full_state_dict.items():
                if k.startswith(moe_prefix):
                    # 切片去掉前缀（不replace，避免proj错位）
                    new_key = k[len(moe_prefix):]  # e.g., "moe.moe.experts.weight" -> "moe.experts.weight"; "moe.proj.weight" -> "proj.weight"
                    moe_subdict[new_key] = v
            
            # 如果空，试备用前缀（无 model.）
            if not moe_subdict:
                alt_prefix = "image_encoder.head.moe."
                for k, v in full_state_dict.items():
                    if k.startswith(alt_prefix):
                        new_key = k[len(alt_prefix):]
                        moe_subdict[new_key] = v
                print(f"[DEBUG] Tried alt prefix '{alt_prefix}', now {len(moe_subdict)} keys")
            
            print(f"[DEBUG] Full MoE subdict keys: {list(moe_subdict.keys())}")  # 打印完整key，确认5个
            print(f"[DEBUG] Filtered MoE subdict: {len(moe_subdict)} keys")
            
            # 加载到 MoE 模块前，打印模型期望的keys
            moe_module = head_layers["moe"]
            expected_keys = list(moe_module.state_dict().keys())
            print(f"[DEBUG] Expected MoE keys: {len(expected_keys)}, sample: {expected_keys[:5]}")
            
            pre_load_sample = moe_module.proj.weight.data[0, 0].item()  # 加载前随机值
            
            # 尝试加载，捕获shape mismatch错误
            try:
                missing, unexpected = moe_module.load_state_dict(moe_subdict, strict=False)
            except RuntimeError as e:
                print(f"[ERROR] Load error (shape mismatch?): {e}")
                missing, unexpected = set(), list(moe_subdict.keys())  # Treat as unexpected
            
            post_load_sample = moe_module.proj.weight.data[0, 0].item()  # 加载后值
            print(f"[DEBUG] MoE load sample: pre={pre_load_sample:.4f} -> post={post_load_sample:.4f} (changed? {pre_load_sample != post_load_sample})")
            
            # 调试信息
            if missing:
                print(f"[WARN] Missing keys: {len(missing)}, sample: {list(missing)[:3]}")
            if unexpected:
                print(f"[WARN] Unexpected keys: {len(unexpected)}, sample: {list(unexpected)[:3]}")
            
            # 如果missing和subdict重叠，可能是key不匹配——打印diff
            missing_list = list(missing)
            subdict_list = list(moe_subdict.keys())
            common = set(missing_list) & set(subdict_list)
            if common:
                print(f"[WARN] Overlap between missing and subdict: {len(common)}, sample: {list(common)[:3]} — key exact match issue?")
            
            # TBD
            self.head_layers = head_layers  # Store dict
            self.proj_layer = self.head_layers.get(proj, None)
            if self.proj_layer is None:
                raise ValueError(f"No head_layer for proj={proj}")
        else:
            raise ValueError(f"Unknown projection type: {proj}")
        
        return nn.Sequential(head_layers)

    def freeze(self, freeze_bn_stats=True):
        """Freeze model params."""
        for param in self.trunk.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)

    def forward(self, x):
        """Forward pass."""
        
    
        features = {}
    
        if self.proj in ["mlp", "linear","moe"]:
            
            x = self.trunk(x)
            
            if self.model_name in ["conch", "h0-mini"]:
                x = x[:, 0, :]
            x = self.head(x)
            
    
        elif self.proj == "transformer":
            
            tokens = self.trunk.forward_features(x)
            x = self.head(tokens)
            x = x[:, 0, :]
            
           
    
        else:
            print(f"[ENCODER DEBUG] No branch matched for proj={self.proj}, returning raw x={x.shape}")
    
        
        return x.contiguous()  # Ensure contiguous memory layout


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
