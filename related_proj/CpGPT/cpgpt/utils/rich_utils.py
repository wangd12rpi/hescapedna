from collections.abc import Sequence
from pathlib import Path
from typing import Any

import rich
import rich.panel
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from cpgpt.data.components.cpgpt_dataset import CpGPTDataset
from cpgpt.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print DictConfig contents as a tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra
        print_order (Sequence[str], optional): Order of config sections to print. Defaults to
            ("data", "model", "callbacks", "logger", "trainer", "paths", "extras")
        resolve (bool, optional): Whether to resolve reference fields. Defaults to False
        save_to_file (bool, optional): Whether to save config to output folder. Defaults to False

    Note:
        Uses Rich library for pretty printing
        Skips fields not found in config with warning

    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing...",
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt user for tags if none provided in config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra
        save_to_file (bool, optional): Whether to save tags to output folder. Defaults to False

    Raises:
        ValueError: If no tags specified before launching multirun

    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            msg = "Specify tags before launching a multirun!"
            raise ValueError(msg)

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


def add_branch(
    branch: Tree,
    original: dict[str, Any],
    current: dict[str, Any],
    path: str = "",
    print_changes_only: bool = False,
    is_original: bool = False,
) -> None:
    """Add branches to tree comparing original and current dictionaries.

    Recursively builds tree structure highlighting parameter changes between
    original and current dictionaries.

    Args:
        branch (Tree): Current tree branch to add elements to
        original (Dict[str, Any]): Original parameters dictionary
        current (Dict[str, Any]): Current parameters dictionary
        path (str, optional): Path to current nested level. Defaults to ""
        print_changes_only (bool, optional): Whether to only show changes. Defaults to False
        is_original (bool, optional): Whether branch represents original dict. Defaults to False

    Note:
        Uses color coding:
        - Cyan for dictionary keys
        - Green for unchanged values
        - Yellow/Red for changed values

    """
    for key, value in current.items():
        current_path = f"{path}.{key}" if path else key
        original_value = original.get(key)

        if isinstance(value, dict) and isinstance(original_value, dict):
            # If the value is a nested dictionary, add a sub-branch and recurse.
            sub_branch = branch.add(f"[bold cyan]{key}[/bold cyan]")
            add_branch(
                sub_branch,
                original_value,
                value,
                current_path,
                print_changes_only,
                is_original,
            )
        else:
            # Determine if the value has changed.
            value_changed = key in original and value != original_value

            # Show only changed values in the original parameters when print_changes_only is True.
            if value_changed:
                if is_original:
                    branch.add(f"[green]{key}:[/green] {original_value!r}")
                else:
                    msg = (
                        f"[yellow]{key}:[/yellow] {value!r} "
                        f"[red](changed from {original_value!r})[/red]"
                    )
                    branch.add(msg)
            elif not print_changes_only:
                branch.add(f"[green]{key}:[/green] {value!r}")


@rank_zero_only
def print_rich_model_info(
    model_info: dict[str, dict[str, Any]],
    print_changes_only: bool = False,
) -> dict[str, Any]:
    """Print model information in formatted tree structure.

    Args:
        model_info (Dict[str, Dict[str, Any]]): Dictionary containing:
            - original_hparams: Original model parameters
            - current_hparams: Current model parameters
        print_changes_only (bool, optional): Whether to only show changes. Defaults to False

    Returns:
        Dict[str, Any]: Current model parameters

    Note:
        Creates panel with two branches:
        - Original parameters (showing only changes if specified)
        - Current parameters (highlighting changes from original)

    """
    tree = Tree("[bold magenta]Model Info[/bold magenta]")

    original_branch = tree.add("[bold cyan]original_hparams[/bold cyan]")
    current_branch = tree.add("[bold cyan]current_hparams[/bold cyan]")

    # Add original parameters, showing only changes if print_changes_only is True
    add_branch(
        original_branch,
        model_info["original_hparams"],
        model_info["current_hparams"],
        print_changes_only=print_changes_only,
        is_original=True,
    )

    # Add current parameters, showing only changes if print_changes_only is True
    add_branch(
        current_branch,
        model_info["original_hparams"],
        model_info["current_hparams"],
        print_changes_only=print_changes_only,
    )

    # Create a panel for the tree and print it
    panel = Panel(
        tree,
        expand=False,
        border_style="bold blue",
        title="[bold red]Model Information[/bold red]",
    )
    rich.print(panel)


@rank_zero_only
def create_rich_dataset_preview(dataset: CpGPTDataset, title: str) -> Panel:
    """Create formatted table displaying dataset information.

    Args:
        dataset (CpGPTDataset): Dataset to display information about
        title (str): Title for the information table

    Returns:
        Panel: Rich panel containing table with:
            - Component names
            - Tensor shapes
            - Data previews
            - Dataset sizes

    Note:
        Color codes different columns:
        - Cyan: Component names
        - Magenta: Shapes
        - Yellow: Previews
        - Green: Dataset sizes

    """
    log.info(f"Creating dataset table: {title}")

    info = Table(title=title)
    info.add_column("Component", style="cyan")
    info.add_column("Shape", style="magenta")
    info.add_column("Preview", style="yellow")
    info.add_column("Dataset Size", style="green")

    # Get the first sample as a dictionary
    sample = dataset[0]
    dataset_size = len(dataset)

    # Define components and their corresponding keys in the dictionary
    components = {
        "Beta values": "meth",
        "DNA embeddings": "dna_embeddings",
        "Chromosomes": "chroms",
        "Positions": "positions",
        "Metadata": "obsm",
    }

    for component_name, key in components.items():
        tensor = sample[key]
        shape = str(tensor.shape) if tensor is not None else "None"

        # Create preview based on component type
        if tensor is not None:
            if key == "dna_embeddings":
                preview = str(tensor[:2, :5].tolist())
            else:
                preview = str(tensor[:5].tolist())
        else:
            preview = "None"

        info.add_row(component_name, shape, preview, str(dataset_size))

    panel = Panel(info, border_style="blue")
    rich.print(panel)
