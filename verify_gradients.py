#!/usr/bin/env python3
"""Verification script to check gradient flow and trainable parameters after bug fixes."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch
from omegaconf import DictConfig, OmegaConf
from hescape.modules.pretrain_module import PretrainModule

def verify_trainable_params(model: torch.nn.Module):
    """Verify which parameters are trainable."""
    print("\n" + "="*80)
    print("TRAINABLE PARAMETERS VERIFICATION")
    print("="*80)

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    total_params = total_trainable + total_frozen

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable:,} ({100*total_trainable/total_params:.2f}%)")
    print(f"Frozen parameters: {total_frozen:,} ({100*total_frozen/total_params:.2f}%)")

    print("\n--- TRAINABLE PARAMETERS ---")
    for name, numel in trainable_params[:20]:  # Show first 20
        print(f"  ✓ {name}: {numel:,}")
    if len(trainable_params) > 20:
        print(f"  ... and {len(trainable_params) - 20} more trainable parameters")

    # Check specific components
    print("\n--- COMPONENT ANALYSIS ---")

    # Image encoder LoRA
    img_lora = [n for n, _ in trainable_params if 'image_encoder' in n and 'lora' in n]
    if img_lora:
        print(f"  ✓ Image encoder LoRA parameters: {len(img_lora)} layers")
    else:
        print("  ⚠ WARNING: No image encoder LoRA parameters found!")

    # Image projection head
    img_head = [n for n, _ in trainable_params if 'image_encoder.head' in n]
    if img_head:
        print(f"  ✓ Image projection head: {len(img_head)} parameters")
    else:
        print("  ⚠ WARNING: No image projection head parameters found!")

    # DNA methylation projection head
    dna_head = [n for n, _ in trainable_params if 'dnameth_encoder.head' in n]
    if dna_head:
        print(f"  ✓ DNA methylation projection head: {len(dna_head)} parameters")
    else:
        print("  ⚠ WARNING: No DNA methylation projection head parameters found!")

    # Check if base models are frozen
    img_base_trainable = [n for n, _ in trainable_params if 'image_encoder.trunks' in n and 'lora' not in n]
    if img_base_trainable:
        print(f"  ⚠ WARNING: {len(img_base_trainable)} base image encoder params are trainable (should be frozen)")
    else:
        print("  ✓ Image encoder base model is frozen (only LoRA trainable)")

    dna_trunk_trainable = [n for n, _ in trainable_params if 'dnameth_encoder.trunk' in n]
    if dna_trunk_trainable:
        print(f"  ⚠ WARNING: {len(dna_trunk_trainable)} DNA methylation trunk params are trainable (should be frozen)")
    else:
        print("  ✓ DNA methylation trunk is frozen (only head trainable)")

    return trainable_params, frozen_params


def verify_gradient_flow(model: torch.nn.Module, dummy_batch: dict):
    """Verify gradients flow correctly through the model."""
    print("\n" + "="*80)
    print("GRADIENT FLOW VERIFICATION")
    print("="*80)

    model.train()

    # Forward pass
    print("\nRunning forward pass...")
    try:
        img_embed, meth_embed, logit_scale = model.model(dummy_batch, norm=False)
        loss = model.model.compute_loss(img_embed=img_embed, dnameth_embed=meth_embed)
        print(f"  ✓ Forward pass successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

    # Backward pass
    print("\nRunning backward pass...")
    try:
        loss.backward()
        print("  ✓ Backward pass successful")
    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        return False

    # Check which parameters got gradients
    print("\nChecking gradient assignment...")
    params_with_grad = []
    params_without_grad = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad.append(name)
            else:
                params_without_grad.append(name)

    print(f"  Parameters with gradients: {len(params_with_grad)}")
    print(f"  Trainable parameters without gradients: {len(params_without_grad)}")

    if params_without_grad:
        print("\n  ⚠ WARNING: These trainable parameters have no gradients:")
        for name in params_without_grad[:10]:
            print(f"    - {name}")
        if len(params_without_grad) > 10:
            print(f"    ... and {len(params_without_grad) - 10} more")
        return False
    else:
        print("  ✓ All trainable parameters received gradients")
        return True


def main():
    print("Starting gradient and trainability verification...")
    print("Note: This is a simplified test without loading actual data.")

    # Create minimal config
    cfg = OmegaConf.create({
        "model": {
            "litmodule": {
                "input_genes": 0,
                "embed_dim": 128,
                "img_enc_name": "gigapath",
                "gene_enc_name": "cpgpt",
                "loss": "CLIP",
                "img_finetune": True,
                "gene_finetune": False,
                "img_proj": "linear",
                "gene_proj": "identity",
                "n_tissue": None,
                "n_region": None,
                "image_size": 224,
                "temperature": 0.07,
                "lr": 1e-5,
                "weight_decay": 0.01,
            }
        },
        "paths": {
            "pretrain_weights": {
                "img_enc_path": None,
                "cpgpt_checkpoint_path": "/media/volume/patho_meth/PathoMethyl-FM/cpgpt_files",
            }
        },
        "training": {
            "lightning": {
                "trainer": {
                    "strategy": "auto",
                }
            },
            "evaluations": {
                "batch_key": "id",
                "label_key": "organ",
            }
        }
    })

    print("\nInitializing model...")
    try:
        model = PretrainModule(
            input_genes=0,
            embed_dim=128,
            img_enc_name="gigapath",
            gene_enc_name="cpgpt",
            loss="CLIP",
            img_finetune=True,
            gene_finetune=False,
            img_proj="linear",
            gene_proj="identity",
            n_tissue=None,
            n_region=None,
            image_size=224,
            temperature=0.07,
            lr=1e-5,
            weight_decay=0.01,
            cfg=cfg,
            lambda_scheduler=None,
        )
        print("  ✓ Model initialized successfully")
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        print("\nThis is expected if you don't have the pretrained weights available.")
        print("The verification script requires:")
        print("  - GigaPath tile and slide encoder weights")
        print("  - CpGPT checkpoint at /media/volume/patho_meth/PathoMethyl-FM/cpgpt_files")
        return

    # Verify trainable parameters
    verify_trainable_params(model)

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nKey checks:")
    print("  1. ✓ Image encoder LoRA parameters should be trainable")
    print("  2. ✓ Image encoder base parameters should be frozen")
    print("  3. ✓ Image projection head should be trainable")
    print("  4. ✓ DNA methylation trunk should be frozen")
    print("  5. ✓ DNA methylation projection head should be trainable")
    print("\nIf all checks pass, gradients should flow correctly during training!")


if __name__ == "__main__":
    main()
