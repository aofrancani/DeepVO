import os
import torch
import torch.optim as optim
from typing import Dict, Optional, Any, Union
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    checkpoint_dict: Dict[str, torch.Tensor],
    checkpoint_fpath: str,
    epoch: int,
    save_best: bool,
    save_interval: int = 5,
) -> None:
    """
    Saves the model checkpoint based on the given parameters.

    Args:
        checkpoint_dict (Dict[str, torch.Tensor]): A dictionary containing the model state, optimizer state,
                                                   scheduler state, and other relevant information to save.
        checkpoint_fpath (str): The directory path where checkpoints should be saved.
        epoch (int): The current epoch number. Used to determine the filename for periodic checkpoints.
        save_best (bool): A flag indicating whether to save the checkpoint as the best model. If `True`,
                          the checkpoint will be saved as "checkpoint_best.pth".
        save_interval (int): The interval in epochs at which checkpoints are saved.
    """
    os.makedirs(checkpoint_fpath, exist_ok=True)

    if save_best:
        torch.save(
            checkpoint_dict, os.path.join(checkpoint_fpath, "checkpoint_best.pth")
        )

    if epoch % save_interval == 0:
        print(f"Saving checkpoint for epoch {epoch} \n")
        torch.save(
            checkpoint_dict, os.path.join(checkpoint_fpath, f"checkpoint_e{epoch}.pth")
        )

    torch.save(checkpoint_dict, os.path.join(checkpoint_fpath, "checkpoint_last.pth"))


def get_optimizer(
    params: torch.nn.Parameter,
    optimizer_args: Optional[Dict[str, Any]] = None,
) -> optim.Optimizer:
    """
    Creates and returns a PyTorch optimizer based on the provided arguments.

    Args:
        params (torch.nn.Parameter): Model parameters to optimize.
        optimizer_args (Optional[Dict[str, Any]]): A dictionary of optimizer settings.
            - "method" (str): The type of optimizer to use.
                Supported methods are "Adam", "SGD", "RAdam", and "Adagrad". Default is "Adam".
            - "lr" (float): Learning rate for the optimizer. Default is 1e-3.
            - "momentum" (float): Momentum factor, only used for SGD. Default is 0.9.
            - "weight_decay" (float): Weight decay (L2 penalty) for the optimizer. Default is 0.0.

    Returns:
        torch.optim.Optimizer: The instantiated optimizer based on the provided method and parameters.

    """
    # Get parameters
    method = optimizer_args.get("method", "Adam")
    learning_rate = optimizer_args.get("lr", 1e-4)
    momentum = optimizer_args.get("momentum", 0.9)
    weight_decay = optimizer_args.get("weight_decay", 1e-4)

    # Build optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)

    elif method == "SGD":
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=learning_rate)

    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: {method}")

    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_args: Optional[Dict[str, Any]] = None,
) -> Optional[_LRScheduler]:
    """
    Creates a learning rate scheduler based on the provided arguments.

    Args:
        optimizer (optim.Optimizer): The optimizer for which to create the scheduler.
        scheduler_args (Optional[Dict[str, Any]]): A dictionary containing scheduler parameters.
            Supported keys include:
                - method (str): The type of scheduler to use. Options: 'StepLR', 'ExponentialLR', 'CosineAnnealingLR', etc.
                - step_size (int): Step size for StepLR.
                - gamma (float): Multiplicative factor for learning rate adjustment in StepLR.
                - T_max (int): Maximum number of iterations for CosineAnnealingLR.
                - eta_min (float): Minimum learning rate for CosineAnnealingLR.

    Returns:
        Optional[_LRScheduler]: The learning rate scheduler instance, or None if no scheduler is specified.
    """

    # Gets scheduler method
    method = scheduler_args.get("method", None)

    if method is None:
        return None

    # Builds scheduler
    if method == "StepLR":
        step_size = scheduler_args.get("step_size", 10)
        gamma = scheduler_args.get("gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif method == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_args.get("milestones", [20, 40, 60]),
            gamma=scheduler_args.get("gamma", 0.5),
        )

    elif method == "ExponentialLR":
        gamma = scheduler_args.get("gamma", 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif method == "CosineAnnealingLR":
        T_max = scheduler_args.get("T_max", 50)
        eta_min = scheduler_args.get("eta_min", 0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    else:
        raise ValueError(f"Unsupported scheduler type: {method}")

    return scheduler


def load_scheduler_state(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint: Union[Dict[str, Any], None],
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Loads the state of the learning rate scheduler from the provided checkpoint.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to load the state into.
        checkpoint (Dict[str, Any]): The checkpoint dictionary containing the scheduler state dict.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Modifies the scheduler in place and returns it.
    """
    if checkpoint:
        try:
            # Load the scheduler state dictionary from the checkpoint
            scheduler.load_state_dict(checkpoint.get("scheduler_state_dict"))
            print("--- Scheduler state loaded successfully.")
        except KeyError:
            print("--- No scheduler state dict found in checkpoint!")
    else:
        print(
            "--- No checkpoint provided. Scheduler initialized with default parameters."
        )

    return scheduler


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: Union[Dict[str, Any], None],
) -> torch.optim.Optimizer:
    """
    Loads the state of the optimizer from the provided checkpoint.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint (Union[Dict[str, Any], None]): The checkpoint dictionary containing the optimizer state dict.
                                                   If None, the optimizer remains in its initialized state.

    Returns:
        torch.optim.Optimizer: The optimizer with the loaded state dict (if checkpoint is provided).
    """
    if checkpoint:
        try:
            # Load the optimizer state dictionary from the checkpoint
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
            print("--- Optimizer state loaded successfully.")
        except KeyError:
            print("--- No optimizer state dict found in checkpoint!")
    else:
        print(
            "--- No checkpoint provided. Optimizer initialized with default parameters."
        )

    return optimizer


def update_training_params(
    training_params: Dict[str, Any],
    checkpoint: Union[Dict[str, Any], None],
) -> Dict[str, Any]:
    """
    Updates the training parameters from the provided checkpoint.

    Args:
        training_params (Dict[str, Any]): A dictionary containing training parameters to be updated.
        checkpoint (Union[Dict[str, Any], None]): The checkpoint dictionary containing training state,
                                                    or None if no checkpoint is provided.

    Returns:
        Dict[str, Any]: The updated training parameters dictionary.

    Raises:
        KeyError: If the epoch or best_val keys are not found in the checkpoint.
    """

    if checkpoint is not None:
        try:
            # Load the last epoch and best validation value
            training_params["epoch_init"] = checkpoint.get("epoch", 1)
            training_params["best_val"] = checkpoint.get("best_val", float("inf"))
            print("--- Training parameters updated successfully.")
        except KeyError as e:
            print(f"--- Error: {e} found in checkpoint!")
    else:
        print("--- No checkpoint provided. Initialized with default parameters.")

    return training_params
