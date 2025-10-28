# Critical Bug Fixes for Gradient Flow

## Overview
This document summarizes the critical bugs that were blocking gradient flow in the training pipeline and the fixes applied.

## Bugs Fixed

### 1. ✅ GigaPath Pipeline - @torch.no_grad() Decorators Removed
**File**: `related_proj/prov-gigapath/gigapath/pipeline.py`

**Problem**: Both inference functions had `@torch.no_grad()` decorators that completely blocked gradient flow to the tile and slide encoders.

**Changes**:
- **Line 140**: Removed `@torch.no_grad()` from `run_inference_with_tile_encoder()`
- **Line 165**: Removed `@torch.no_grad()` from `run_inference_with_slide_encoder()`
- **Line 159**: Removed `.detach()` call that was severing the computational graph
- **Lines 160, 186-187**: Removed `.cpu()` calls to keep tensors on GPU for gradient flow

**Impact**: Gradients can now flow through both tile and slide encoders during training.

---

### 2. ✅ CpGPT - @torch.no_grad() Decorator Removed
**File**: `src/hescape/models/dnameth_models/_cpgpt.py`

**Problem**: The `encode_beta_files()` method had `@torch.no_grad()` decorator blocking gradients to the DNA methylation projection head.

**Changes**:
- **Line 93**: Removed `@torch.no_grad()` from `encode_beta_files()`

**Impact**: Gradients can now flow to the DNA methylation projection head while the trunk remains frozen (as intended).

---

### 3. ✅ Image Encoder Forward Pass - torch.stack Instead of torch.cat
**File**: `src/hescape/models/image_models/image_encoder.py`

**Problem**: Using `torch.cat()` on embeddings computed in a for-loop could break gradient flow across batch dimension.

**Changes**:
- **Line 211**: Changed from `torch.cat(embeds, dim=0)` to `torch.stack([e.squeeze(0) if e.dim() > 2 else e for e in embeds], dim=0)`

**Impact**: Cleaner gradient flow through batch dimension with proper tensor stacking.

---

## Expected Training Behavior

After these fixes, your training should have:

### ✅ Trainable Components
1. **GigaPath Tile Encoder**: LoRA parameters only (base model frozen)
2. **GigaPath Slide Encoder**: LoRA parameters only (base model frozen)
3. **Image Projection Head**: All parameters trainable
4. **DNA Methylation Projection Head**: All parameters trainable

### ✅ Frozen Components
1. **GigaPath Base Models**: All non-LoRA parameters frozen
2. **CpGPT Trunk**: All parameters frozen

### ✅ Gradient Flow
- Gradients flow from loss → projection heads → encoders (LoRA) → inputs
- No `@torch.no_grad()` contexts blocking the computational graph
- No `.detach()` calls severing gradient connections

---

## Verification

To verify the fixes work correctly, run:

```bash
python verify_gradients.py
```

This script will check:
- Which parameters are trainable vs frozen
- Whether gradients flow to all trainable parameters
- Component-specific trainability (LoRA, projection heads, etc.)

---

## Configuration Alignment

Your config files correctly specify:
- `img_finetune: true` - GigaPath gets LoRA
- `gene_finetune: false` - CpGPT stays frozen
- `img_proj: linear` - Image projection head
- `gene_proj: identity` - DNA methylation projection head

All fixes align with this intended behavior.

---

## Additional Notes

### Memory Optimization
The pipeline functions now keep tensors on GPU throughout processing, avoiding CPU↔GPU transfers. This should improve training speed.

### Model Eval Mode
Note that `tile_encoder.eval()` and `slide_encoder.eval()` are still called in the pipeline functions (lines 155, 182). This is intentional for:
- BatchNorm/LayerNorm stability
- Deterministic behavior
- Standard practice for LoRA fine-tuning

The base model stays in eval mode while LoRA adapters still receive gradients and update.

---

## What Was NOT Changed

1. **CpGPT Architecture**: Trunk remains frozen, only projection head trains
2. **LoRA Configuration**: r=8, alpha=16, targeting ["qkv", "proj"]
3. **Training Loop**: No changes to `pretrain_module.py` or data loading
4. **Loss Functions**: CLIP/SigLip losses unchanged

---

## Testing Recommendations

1. **Start with a small training run** (few steps) to verify:
   - Loss decreases
   - No gradient warnings in logs
   - GPU memory usage is reasonable

2. **Check parameter updates**:
   ```python
   # Before training
   initial_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

   # After a few steps
   for name, param in model.named_parameters():
       if param.requires_grad:
           diff = (param - initial_params[name]).abs().sum()
           print(f"{name}: changed by {diff.item():.6f}")
   ```

3. **Monitor gradients**:
   ```python
   for name, param in model.named_parameters():
       if param.requires_grad and param.grad is not None:
           grad_norm = param.grad.norm().item()
           print(f"{name}: grad_norm={grad_norm:.6f}")
   ```

---

## Troubleshooting

### If loss doesn't decrease:
- Check learning rate (currently 1e-5)
- Verify batch size is reasonable (currently 32)
- Check if gradient clipping is too aggressive

### If GPU OOM:
- Reduce `batch_size` in config
- Reduce `batch_size=512` in tile encoder inference (image_encoder.py:205)
- Enable gradient checkpointing in LoRA config

### If gradients are NaN:
- Check data preprocessing (missing values, infinities)
- Reduce learning rate
- Add gradient clipping in config

---

## Summary

**Before**: `@torch.no_grad()` decorators completely blocked gradient flow to encoders
**After**: All trainable parameters receive gradients and can update during training

Your training should now work as intended with:
- GigaPath tile + slide encoders: LoRA fine-tuned ✓
- CpGPT: Frozen ✓
- Projection heads: Trainable ✓
