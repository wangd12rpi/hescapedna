import os
import warnings
from pathlib import Path
from typing import Any

# os.environ["NCCL_P2P_LEVEL"] = "PIX"
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OmegaConf.register_new_resolver("eval", eval)
import faulthandler

from pytorch_lightning import seed_everything

faulthandler.enable()


def _safe_len(obj: Any) -> int | None:
    try:
        return len(obj)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _loader_stats(loader: Any) -> dict[str, int | None]:
    stats = {"batches": _safe_len(loader), "samples": None}
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        stats["samples"] = _safe_len(dataset)
    return stats


def _summarize_lora(image_encoder: Any, finetune_enabled: bool) -> tuple[str, list[str]]:
    trunk = getattr(image_encoder, "trunk", None)
    if trunk is None:
        return "unknown (missing trunk)", []

    details: list[str] = []
    if not finetune_enabled:
        return "frozen (no fine-tuning)", details

    peft_config = getattr(trunk, "peft_config", None)
    if peft_config:
        method = "LoRA"
        for name, cfg in peft_config.items():
            cfg_fields = []
            peft_type = getattr(cfg, "peft_type", None)
            if peft_type:
                cfg_fields.append(f"peft_type={peft_type}")
            rank = getattr(cfg, "r", None)
            if isinstance(rank, list):
                method = "MoE-LoRA"
            if rank is not None:
                cfg_fields.append(f"r={rank}")
            alpha = getattr(cfg, "lora_alpha", None)
            if alpha is not None:
                cfg_fields.append(f"alpha={alpha}")
            targets = getattr(cfg, "target_modules", None)
            if targets is not None:
                cfg_fields.append(f"targets={targets}")
            num_experts = getattr(cfg, "num_experts", None)
            if num_experts:
                method = "MoE-LoRA"
                cfg_fields.append(f"num_experts={num_experts}")
            top_k = getattr(cfg, "top_k", None)
            if top_k:
                cfg_fields.append(f"top_k={top_k}")
            details.append(f"adapter '{name}': " + ", ".join(str(item) for item in cfg_fields))
        if not details:
            details.append("PeftModel detected but no adapters registered.")
        return method, details

    trainable = sum(1 for p in trunk.parameters() if p.requires_grad)
    total = sum(1 for _ in trunk.parameters())
    if total == 0:
        return "unknown (no parameters)", details
    if trainable == 0:
        return "frozen (no trainable trunk params)", details
    if trainable == total:
        return "full fine-tune", details
    return f"partial fine-tune ({trainable}/{total} params trainable)", details


def _log_training_overview(
    hescape_logger: Any,
    cfg: DictConfig,
    train_loader: Any,
    valid_loader: Any,
    test_loader: Any,
    lightning_module: Any,
) -> None:
    stats = {
        "train": _loader_stats(train_loader),
        "val": _loader_stats(valid_loader),
        "test": _loader_stats(test_loader),
    }
    fmt = lambda value: "?" if value is None else str(value)
    hescape_logger.info(
        "Dataset splits -> train: %s samples / %s batches | val: %s / %s | test: %s / %s",
        fmt(stats["train"]["samples"]),
        fmt(stats["train"]["batches"]),
        fmt(stats["val"]["samples"]),
        fmt(stats["val"]["batches"]),
        fmt(stats["test"]["samples"]),
        fmt(stats["test"]["batches"]),
    )

    total_params = sum(p.numel() for p in lightning_module.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    hescape_logger.info(
        "Trainable parameters: %s out of %s (%.2f%%)",
        trainable_params,
        total_params,
        (trainable_params / total_params * 100) if total_params else 0.0,
    )

    image_encoder = getattr(lightning_module.model, "image_encoder", None)
    if image_encoder is not None:
        method, adapter_details = _summarize_lora(image_encoder, cfg.model.litmodule.img_finetune)
        hescape_logger.info(
            "Image encoder '%s' | finetune=%s | projection=%s | slide_encoder=%s",
            cfg.model.litmodule.img_enc_name,
            cfg.model.litmodule.img_finetune,
            cfg.model.litmodule.img_proj,
            "yes" if getattr(image_encoder, "use_slide_encoder", False) else "no",
        )
        hescape_logger.info("  Adaptation strategy: %s", method)
        for detail in adapter_details:
            hescape_logger.info("    %s", detail)
        if not adapter_details:
            hescape_logger.info("    No adapter-specific configuration detected.")
    else:
        hescape_logger.info("Image encoder summary unavailable (module missing).")

    gene_encoder = getattr(lightning_module.model, "dnameth_encoder", None)
    if gene_encoder is not None:
        hescape_logger.info(
            "DNA methylation encoder '%s' | finetune=%s | projection=%s",
            cfg.model.litmodule.gene_enc_name,
            cfg.model.litmodule.gene_finetune,
            cfg.model.litmodule.gene_proj,
        )
    else:
        hescape_logger.info("DNA methylation encoder summary unavailable (module missing).")

    hescape_logger.info(
        "Loss=%s | temperature=%.3f | optimizer.lr=%s | weight_decay=%s",
        cfg.model.litmodule.loss,
        cfg.model.litmodule.temperature,
        cfg.model.optimizer.lr,
        cfg.model.optimizer.weight_decay,
    )


def train(cfg: DictConfig) -> None:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
    from pytorch_lightning.loggers import Logger

    from hescape.modules.pretrain_module import ClampCallback

    torch.set_float32_matmul_precision("medium")
    from hescape._logging import logger as hescape_logger

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        hescape_logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            hescape_logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        hescape_logger.info("No GPUs available.")

    if cfg.datamodule.get("seed"):
        pl.seed_everything(cfg.datamodule.seed, workers=True)

    hescape_logger.info(f"Instantiating datamodule {cfg.datamodule}")
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()

    # For DNA methylation: dynamically get input_genes after setup
    if hasattr(dm, 'input_genes'):
        input_genes = dm.input_genes
        hescape_logger.info(f"DNA methylation: Using {input_genes} CpG sites")
        # Update config with actual input_genes
        cfg.model.litmodule.input_genes = input_genes

    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # hescape_logger.info("Instantiating model...")
    lr_lambda = hydra.utils.instantiate(cfg.model.cosine_scheduler)
    model: LightningModule = hydra.utils.instantiate(cfg.model.litmodule)
    model = model(lambda_scheduler=lr_lambda, cfg=cfg)
    _log_training_overview(hescape_logger, cfg, train_loader, valid_loader, test_loader, model)

    hescape_logger.info("Instantiating callbacks and logger...")
    callbacks: list[Callback] = []
    for _, cb in cfg.training.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb))
    callbacks.append(ClampCallback())

    logger: list[Logger] = []
    for name, lg in cfg.training.logger.items():
        lgr = hydra.utils.instantiate(lg)

        logger.append(lgr)

    hescape_logger.info("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(cfg.training.lightning.trainer)
    trainer = trainer(callbacks=callbacks, logger=logger,  num_sanity_val_steps=0)

    if cfg.training.train:
        hescape_logger.info("Training...")

        trainer.fit(
            model,
            # datamodule=dm,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
        if trainer.global_rank == 0:
            hescape_logger.info("Testing...")
            trainer.test(
                model,
                verbose=True,
                dataloaders=test_loader,
            )


def find_project_root(path: str = ".") -> str:
    """Recursively finds the project root by looking for a '.git' directory."""
    # Start from the current working directory
    current_dir = Path(os.getcwd()).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / ".git").is_dir():
            return str(current_dir)
        current_dir = current_dir.parent
    # If .git is not found, raise an error
    raise FileNotFoundError("Could not find project root with .git directory.")


# Register the custom resolver with OmegaConf
OmegaConf.register_new_resolver("project_root", find_project_root)


@hydra.main(config_path="./../configs", config_name="local_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    from rich.pretty import pprint

    sweep_params = HydraConfig.get().job.override_dirname

    img_enc_name = cfg.model.litmodule.img_enc_name
    dnameth_enc_name = cfg.model.litmodule.gene_enc_name
    img_proj = cfg.model.litmodule.img_proj
    img_finetune = cfg.model.litmodule.img_finetune
    dnameth_proj = cfg.model.litmodule.gene_proj
    dnameth_finetune = cfg.model.litmodule.gene_finetune

    seed = cfg.datamodule.seed
    # batch_size = cfg.datamodule.batch_size
    loss = cfg.model.litmodule.loss

    # set seed for reproducibility
    seed_everything(seed, workers=True)

    try:
        job_id = f"{HydraConfig.get().job.id}"  # _{HydraConfig.get().job.num}"
    except Exception:
        job_id = "local"
    wandb_name = f"{job_id}"
    cfg.training.logger.wandb.name = wandb_name
    cfg.paths.anatomy.output = f"{cfg.paths.anatomy.output}/{wandb_name}"

    pprint(OmegaConf.to_container(cfg, resolve=True))
    train(cfg)

    print(f"SWEEP PARAMS {sweep_params}, {cfg.paths.anatomy.output}")
    print(f"modelcheckpoint dirpath: {cfg.training.callbacks.model_checkpoint.dirpath}")
    print(f"csv logger name: {cfg.training.logger.csv.save_dir}")
    # print(f"encoder_path: {cfg.model.litmodule.encoder_path}")

    # os.makedirs(cfg.training.logger.wandb.save_dir, exist_ok=True)


if __name__ == "__main__":
    main()
