# Project description

## Introduction

- Original hescape code aligned slide images with gene expression.
- We now align tiled slide images with DNA methylation beta values (paired 1 slide : 1 beta table), so the gene-expression pipeline has been removed.
- This repo keeps the MoE/LoRA image encoder additions from the fork while simplifying the CpGPT integration for methylation.

## Aligning 2 modalities

1. Tiled slide images → Gigapath → projection head.
2. Beta-value TSV → CpGPT runner (filters to vocab, drops NaNs) → optional projection head.

Any encoders including tile encoder, slide encoder, dna encoder might be finetuned using peft or frozen.

Note MoE shouldn't be on the projection both sides as it might impact training, for now only injecting LoRA on image encoder, not on cpgpt.

## Models
We are using Gigapath for images and CpGPT for DNA methylation; the structure still allows other encoders if needed later.

## Environement

- `conda activate gigapath`
