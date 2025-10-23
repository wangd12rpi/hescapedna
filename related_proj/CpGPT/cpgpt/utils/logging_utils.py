import sys
from pathlib import Path
from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from cpgpt.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Log hyperparameters using Lightning loggers.

    Controls which configuration parts are saved by loggers and adds additional
    metrics like model parameter counts.

    Args:
        object_dict (Dict[str, Any]): Dictionary containing:
            - cfg: DictConfig with main configuration
            - model: Lightning model instance
            - trainer: Lightning trainer instance

    Note:
        Logs the following sections:
        - Model configuration and parameters
        - Data configuration
        - Trainer settings
        - Callbacks
        - Extra configurations
        - Task name, tags, checkpoint path, and seed

    Raises:
        Warning: If no logger is found in trainer

    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["trainer_ckpt_path"] = cfg.get("trainer_ckpt_path")
    hparams["model_ckpt_path"] = cfg.get("model_ckpt_path")
    hparams["strict_load"] = cfg.get("strict_load")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for log_instance in trainer.loggers:
        log_instance.log_hyperparams(hparams)


def setup_logger(script_name: str, log_level: str = "INFO"):
    """Sets up a logger with customized formatting and output destinations.

    This function configures a logger with a specific format and directs output to both
    stdout and a rotating log file. It creates a logs directory if it doesn't exist.

    Args:
        script_name (str): Name of the script, used for the log filename.
        log_level (str, optional): Logging level threshold. Defaults to "INFO".

    Returns:
        Logger: Configured logger instance ready for use.

    Example:
        >>> logger = setup_logger("my_script")
        >>> logger.info("This is an info message")

    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    log_file = f"logs/{script_name}.log"

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.remove()
    logger.configure(extra={"class_name": "MainScript"})
    logger.add(sys.stdout, level=log_level, format=log_format)
    logger.add(log_file, level=log_level, format=log_format, rotation="10 MB")

    return logger
