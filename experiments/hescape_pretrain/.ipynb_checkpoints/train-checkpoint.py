import os
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

# --- Pre-amble and Setup ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
OmegaConf.register_new_resolver("eval", eval)

def find_project_root(path: str = ".") -> str:
    current_dir = Path(os.getcwd()).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / ".git").is_dir():
            return str(current_dir)
        current_dir = current_dir.parent
    return "/home/exouser/Public/modified/experiments/"
OmegaConf.register_new_resolver("project_root", find_project_root)


# --- Test Function (no changes here) ---
def test(cfg: DictConfig, ckpt_path: str) -> None:
    # ... (The test function remains exactly the same as before)
    # It correctly receives cfg and ckpt_path
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer
    from pytorch_lightning.loggers import Logger
    from hescape._logging import logger as hescape_logger

    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        hescape_logger.info(f"Using {torch.cuda.device_count()} GPU(s).")
    else:
        hescape_logger.info("No GPUs available, using CPU.")

    if cfg.datamodule.get("seed"):
        pl.seed_everything(cfg.datamodule.seed, workers=True)

    hescape_logger.info("Instantiating datamodule...")
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    test_loader = dm.test_dataloader()
    hescape_logger.info(f"Test dataset contains {len(test_loader.dataset)} samples.")

    hescape_logger.info("Instantiating model architecture...")
    lr_lambda = hydra.utils.instantiate(cfg.model.cosine_scheduler)
    model: LightningModule = hydra.utils.instantiate(cfg.model.litmodule)
    model = model(lambda_scheduler=lr_lambda, cfg=cfg)

    hescape_logger.info("Instantiating logger...")
    logger: list[Logger] = []
    if cfg.training.get("logger"):
        for _, lg_conf in cfg.training.logger.items():
            logger.append(hydra.utils.instantiate(lg_conf))

    hescape_logger.info("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.training.lightning.trainer,
        logger=logger,
        callbacks=None
    )

    hescape_logger.info(f"Starting testing with checkpoint: {ckpt_path}")
    if not Path(ckpt_path).is_file():
        hescape_logger.error(f"Checkpoint file not found at: {ckpt_path}")
        return

    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path=ckpt_path,
        verbose=True
    )
    hescape_logger.info("Testing finished.")


# --- Main Entry Point ---

# 1. CHANGED: config_path is now relative. This makes the script portable.
@hydra.main(config_path="../configs", config_name="local_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that loads config and initiates the test.
    The paths can now be overridden from the command line.
    """
    from rich.pretty import pprint

    # 2. CHANGED: Read the checkpoint path from the config object.
    # This allows it to be set from the command line.
    # We check if 'test_ckpt_path' exists in the config. If not, we raise an error.
    if "test_ckpt_path" not in cfg:
        raise ValueError("Checkpoint path must be provided. \nUsage: python test.py test_ckpt_path=/path/to/your.ckpt")
    
    ckpt_path = cfg.test_ckpt_path

    print("--- Using the following configuration for testing ---")
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print("----------------------------------------------------")

    test(cfg, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()