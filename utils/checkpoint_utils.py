from typing import Dict, Union, Optional, Tuple
from pathlib import Path

import torch

from models.build_model import load_model_state
from utils.train_utils import (
    load_optimizer_state,
    load_scheduler_state,
    update_training_params,
)


def load_checkpoint(checkpoint_fpath: Union[str, Path, None]) -> Optional[Dict]:
    """
    Loads a PyTorch model checkpoint into a specified device.

    Args:
        checkpoint_fpath (str): The file path of the checkpoint. If ".pth" is not present, it will be added.

    Returns:
        Dict: Checkpoint dictionary
    """

    # No checkpoint
    if checkpoint_fpath is None:
        return None

    # Convert checkpoint_fpath to string
    checkpoint_fpath = str(checkpoint_fpath)

    # Ensure ".pth" is part of the checkpoint file path
    if not checkpoint_fpath.endswith(".pth"):
        checkpoint_fpath += ".pth"

    # Load checkpoint and the model's state dict
    checkpoint = torch.load(checkpoint_fpath)

    return checkpoint


def define_initialization_checkpoint(
    checkpoint_params: Union[Path, str]
) -> Union[Path, None]:
    """
    Determines the initialization checkpoint for the model based on the provided checkpoint parameters.
    It checks whether to restart training from the last checkpoint, initialize from a given model
    file, or start training from scratch.

    Args:
        checkpoint_params (Union[Path, str]): Dictionary with checkpoint parameters. It should contain:
            - "checkpoint_dpath" (str): The directory path where checkpoints are stored.
            - "model_init_fpath" (str, optional): The file path of the initial model to load. If not provided,
              the model will be initialized from scratch or the last checkpoint.

    Returns:
        Union[Path, None]: The file path of the checkpoint to initialize the model.
                           Returns None if no checkpoint is available.
    """

    # Paths from arguments
    checkpoint_dpath = Path(checkpoint_params["checkpoint_dpath"])
    checkpoint_last_fpath = checkpoint_dpath / "checkpoint_last.pth"
    model_init_fpath = checkpoint_params.get("model_init_fpath", None)

    # Ensure ".pth" is part of the checkpoint file path
    if model_init_fpath and not model_init_fpath.endswith(".pth"):
        model_init_fpath += ".pth"

    # Case 1: No initialization found: Restart training from last checkpoint if it exists
    if checkpoint_last_fpath.is_file():
        print(f"-- Restarting training from last checkpoint found")
        checkpoint_fpath = checkpoint_last_fpath

    # Case 2: No last checkpoint found and initialization fpath is given and exists
    elif model_init_fpath and Path(model_init_fpath).is_file():
        print(f"-- Initializing model with {model_init_fpath}")
        checkpoint_fpath = Path(model_init_fpath)

    # Case 3: No initialization (from scratch)
    else:
        checkpoint_fpath = None

    return checkpoint_fpath


def update_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_params: Union[Dict[str, str], str],
    training_params: Union[Dict[str, str], str],
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Dict
]:
    """
    Updates the model, optimizer, and scheduler states from a checkpoint.

    Args:
        model (torch.nn.Module): The model whose state will be updated.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be updated.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler whose state will be updated.
        checkpoint_params (Union[Dict[str, str], str]): The parameters for loading the checkpoint.
        training_params (Union[Dict[str, str], str]): The training initialization parameters.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Dict]:
            The updated model, optimizer, and scheduler and training dict after loading the checkpoint state.
    """

    # Define the checkpoint file path
    checkpoint_fpath = define_initialization_checkpoint(checkpoint_params)

    if checkpoint_fpath is None:
        return model, optimizer, scheduler, training_params

    # Check if it is a new training or restarting
    restart_training = False
    if checkpoint_fpath and ("checkpoint_last" in checkpoint_fpath.name):
        restart_training = True

    # Load the checkpoint
    checkpoint = load_checkpoint(checkpoint_fpath)

    # Update model state
    model = load_model_state(model, checkpoint)

    if restart_training:
        # Update optimizer state
        optimizer = load_optimizer_state(optimizer, checkpoint)

        # Update scheduler state
        scheduler = load_scheduler_state(scheduler, checkpoint)

        # Update training parameters
        training_params = update_training_params(training_params, checkpoint)

    return model, optimizer, scheduler, training_params
