import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from cpgpt.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiate callbacks from configuration.

    Creates Lightning callback objects based on provided configurations.

    Args:
        callbacks_cfg (DictConfig): Configuration containing callback specifications

    Returns:
        List[Callback]: List of instantiated Lightning callbacks

    Raises:
        TypeError: If callbacks_cfg is not a DictConfig

    Note:
        Only instantiates callbacks that have a "_target_" field specified

    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        msg = "Callbacks config must be a DictConfig!"
        raise TypeError(msg)

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from configuration.

    Creates Lightning logger objects based on provided configurations.

    Args:
        logger_cfg (DictConfig): Configuration containing logger specifications

    Returns:
        List[Logger]: List of instantiated Lightning loggers

    Raises:
        TypeError: If logger_cfg is not a DictConfig

    Note:
        Only instantiates loggers that have a "_target_" field specified

    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        msg = "Logger config must be a DictConfig!"
        raise TypeError(msg)

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
