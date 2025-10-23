import warnings
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

from omegaconf import DictConfig

from cpgpt.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before task execution.

    Applies various optional utilities including:
    - Python warning suppression
    - Command-line tag enforcement
    - Rich config printing

    Args:
        cfg (DictConfig): Configuration object containing the config tree

    Note:
        Utilities are only applied if corresponding flags are set in cfg.extras

    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Decorate task function to handle execution failures gracefully.

    Provides wrapper functionality for:
    - Ensuring logger cleanup on exceptions
    - Saving exceptions to log files
    - Marking failed runs
    - Closing wandb runs properly

    Args:
        task_func (Callable): The task function to wrap

    Returns:
        Callable: Wrapped task function

    Example:
        ```python
        @task_wrapper
        def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            return metric_dict, object_dict
        ```

    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieve metric value from logging dictionary.

    Args:
        metric_dict (Dict[str, Any]): Dictionary containing metric values
        metric_name (Optional[str]): Name of metric to retrieve

    Returns:
        Optional[float]: Value of specified metric if found, None if metric_name is None

    Raises:
        Exception: If metric_name not found in metric_dict

    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        msg = (
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
        raise Exception(
            msg,
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
