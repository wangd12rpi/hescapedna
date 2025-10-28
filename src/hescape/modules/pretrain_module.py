from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from hescape._logging import logger
from hescape.models import CLIPModel


def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class ClampCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.model.logit_scale.clamp_(min=0, max=math.log(100))


class PretrainModule(LightningModule):
    def __init__(
        self,
        input_genes: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "uni", "conch", "optimus", "densenet", "gigapath"],
        gene_enc_name: str,
        loss: Literal["CLIP", "SIGLIP"],
        tile_finetune: bool,
        slide_finetune: bool,
        gene_finetune: bool,
        img_proj: Literal["linear", "mlp", "transformer"],
        gene_proj: Literal["identity", "linear", "mlp"],
        n_tissue: int,
        n_region: int,
        image_size: int,
        temperature: float,
        lr: float,
        weight_decay: float,
        cfg: DictConfig,
        lambda_scheduler: Callable | None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        local_rank, global_rank, world_size = world_info_from_env()

        if torch.cuda.is_available():
            logger.info(f"CUDA DEVICE NAME: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"CUDA DEVICE INDEX: {torch.cuda.current_device()}")
            logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")
        else:
            logger.info("CUDA is not available.")

        # Determine DNA methylation encoder path
        dnameth_enc_path = getattr(self.cfg.paths.pretrain_weights, "gene_enc_path", None)
        if hasattr(self.cfg.paths.pretrain_weights, "cpgpt_checkpoint_path"):
            dnameth_enc_path = self.cfg.paths.pretrain_weights.cpgpt_checkpoint_path

        self.model = CLIPModel(
            input_sites=input_genes,
            embed_dim=embed_dim,
            img_enc_name=img_enc_name,
            loss=loss,
            tile_finetune=tile_finetune,
            slide_finetune=slide_finetune,
            dnameth_finetune=gene_finetune,
            dnameth_enc_name=gene_enc_name,
            image_size=image_size,
            n_tissue=n_tissue,
            n_region=n_region,
            temperature=temperature,
            world_size=world_size,
            rank=local_rank,
            img_enc_path=self.cfg.paths.pretrain_weights.img_enc_path,
            dnameth_enc_path=dnameth_enc_path,
            drvi_model_dir=self.cfg.paths.anatomy.pretrain_weights.drvi_model_dir if hasattr(self.cfg.paths, "anatomy") else None,
            cfg=cfg,
            img_proj=img_proj,
            dnameth_proj=gene_proj,
        )
        self.lambda_scheduler: Callable[[int], float] | None = lambda_scheduler
        self.lr = lr
        self.weight_decay = weight_decay

        # evaluations
        self.eval_outputs = {"val": [], "test": []}
        self.strategy = self.cfg.training.lightning.trainer.strategy
        self.eval_batch_key = self.cfg.training.evaluations.batch_key
        self.eval_label_key = self.cfg.training.evaluations.label_key

    def configure_optimizers(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)
        print("Creating AdamW with betas =", (0.9, 0.95))
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.95), lr=self.lr, weight_decay=self.weight_decay)
        if self.lambda_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lambda_scheduler)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def _get_moe_aux_loss(self) -> torch.Tensor:
        aux_total = torch.zeros((), device=self.device)
        for enc in [getattr(self.model, "image_encoder", None), getattr(self.model, "dnameth_encoder", None)]:
            if enc is None:
                continue
            for m in enc.head.modules():
                if hasattr(m, "last_aux_loss"):
                    val = m.last_aux_loss
                    if not torch.is_tensor(val):
                        val = torch.as_tensor(val, device=self.device, dtype=torch.float32)
                    aux_total = aux_total + val
        return aux_total

    def _batch_size(self, batch: dict[str, Any]) -> int:
        """Infer batch size for image inputs that may be tensors or slide path lists."""
        images = batch.get("image")
        if images is None:
            raise KeyError("Batch missing 'image' key; cannot infer batch size.")
        if isinstance(images, torch.Tensor):
            return images.shape[0]
        if isinstance(images, (list, tuple)):
            return len(images)
        if hasattr(images, "__len__"):
            return len(images)  # fall back to generic containers (e.g., numpy arrays)
        raise TypeError(f"Unsupported image batch type {type(images)!r} for size inference.")

    def training_step(self, batch, batch_idx):
        bs = self._batch_size(batch)
        main_loss, metrics = self.shared_step(batch, "train")

        aux_moe = self._get_moe_aux_loss()
        loss = main_loss + aux_moe

        metrics["train/aux_moe"] = aux_moe
        metrics["train/loss"] = loss
        metrics["train/main_loss"] = main_loss

        logit_bias = getattr(self.model, "logit_bias", None)
        if logit_bias is not None:
            metrics["logit_bias"] = logit_bias.item()
        metrics["logit_scale"] = self.model.logit_scale.item()

        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return loss


    def validation_step(self, batch, batch_idx):
        bs = self._batch_size(batch)
        _, metrics = self.shared_step(batch, "val")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics

    def test_step(self, batch, batch_idx):
        bs = self._batch_size(batch)
        _, metrics = self.shared_step(batch, "test")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics

    def shared_step(self, batch, stage: str):
        img_embed, meth_embed, logit_scale = self.model(batch, norm=False)
        # For batch "ID" and "GEXP"
        # source_ids = batch[DatasetEnum.ID].to(torch.int64)
        # source_exp = batch[DatasetEnum.GEXP].to(img_embed.dtype)

        contrastive_loss = self.model.compute_loss(img_embed=img_embed, dnameth_embed=meth_embed)

        metrics = {f"{stage}_loss": contrastive_loss}

        knn_recall_i2g = get_clip_metrics(img_embed, meth_embed, logit_scale=logit_scale, stage=stage)
        metrics.update(knn_recall_i2g)

        return contrastive_loss, metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)


def get_clip_metrics(image_features, dnameth_features, logit_scale, stage: str):
    metrics = {}
    logits_per_image = logit_scale * image_features @ dnameth_features.T
    logits_per_dnameth = logits_per_image.T

    logits = {"image_to_dnameth": logits_per_image, "dnameth_to_image": logits_per_dnameth}
    ground_truth = torch.arange(len(dnameth_features), device=dnameth_features.device).contiguous().view(-1, 1)

    # metrics for +ve and -ve pairs
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.float()
        metrics[f"{stage}/{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{stage}/{name}_median_rank"] = torch.floor(torch.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{stage}/{name}_R@{k}"] = torch.mean((preds < k).float())

    return metrics
