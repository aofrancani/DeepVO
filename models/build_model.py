from typing import Any, Dict, Union

import torch
from models.deepvo import DeepVO


def build_model(
    model_params: Dict[str, Any], device: Union[torch.device, str]
) -> torch.nn.Module:
    """
    Builds a DeepVO model based on the provided parameters.

    Args:
        model_params (Dict[str, Any]): A dictionary containing the parameters needed to build the model.
        device (Union[torch.device, str]): Either a CPU or a CUDA device.

    Returns:
        torch.nn.Module: DeepVO model.
    """

    # Build model
    model = DeepVO(
        input_channels=model_params["input_channels"],
        hidden_size=model_params["hidden_size"],
        lstm_layers=model_params["lstm_layers"],
        output_size=model_params["output_size"],
        lstm_dropout=model_params["lstm_dropout"],
    )

    # Send model to device
    model = model.to(device)

    return model


def load_model_state(
    model: torch.nn.Module,
    checkpoint: Union[Dict[str, Any], None],
) -> torch.nn.Module:
    """
    Loads the state dict of the model from a given checkpoint.

    Args:
        model (torch.nn.Module): The model to load the state into.
        checkpoint (Union[Dict[str, Any], None]): The checkpoint dictionary containing the model's state dict.
                                                   If None, the model remains in its initialized state.

    Returns:
        torch.nn.Module: The model with the loaded state dict (if checkpoint is provided),
                         or the model initialized with default parameters.
    """
    if checkpoint:
        try:
            # Load the model state dictionary from the checkpoint
            model.load_state_dict(checkpoint.get("model_state_dict"))
            print("--- Model state loaded successfully.")
        except KeyError:
            print("--- No model state dict found in checkpoint!")
    else:
        print("--- No checkpoint provided. Model initialized with default parameters.")

    return model


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the given model.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
