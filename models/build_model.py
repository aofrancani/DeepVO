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
        input_res=model_params["image_size"],
        hidden_size=model_params["hidden_size"],
        lstm_layers=model_params["lstm_layers"],
        output_size=model_params["output_size"],
        lstm_dropout=model_params["lstm_dropout"],
        conv_dropout=model_params["conv_dropout"],
    )

    # Load pretrained FlowNet
    if model_params.get("pretrained_flownet", False):
        flownet_checkpoint_fpath = model_params.get(
            "flownet_checkpoint", "checkpoints/flownet/flownets_from_caffe.pth"
        )
        model = load_pretrained_flownet(model, flownet_checkpoint_fpath)

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


def load_pretrained_flownet(
    deepvo_model: torch.nn.Module,
    flownet_checkpoint_fpath: str = "checkpoints/flownet/flownets_from_caffe.pth",
):
    """
    Load FlowNet encoder weights into the DeepVO model's feature extractor layers.

    Args:
        deepvo_model (torch.nn.Module): The DeepVO model instance.
        flownet_checkpoint_fpath (str): Path to the FlowNet checkpoint file.

    Returns:
        torch.nn.Module: The DeepVO model with encoder layers initialized from FlowNet.
    """
    # Load FlowNet checkpoint
    flownet_checkpoint = torch.load(flownet_checkpoint_fpath, weights_only=True)
    flownet_state_dict = flownet_checkpoint["state_dict"]

    # Get current DeepVO model state dict
    deepvo_state_dict = deepvo_model.state_dict()

    # Map FlowNet layer names to DeepVO feature extractor layer names
    mapping = {
        "conv1.0": "feature_extractor.conv1",
        "conv2.0": "feature_extractor.conv2",
        "conv3.0": "feature_extractor.conv3",
        "conv3_1.0": "feature_extractor.conv3_1",
        "conv4.0": "feature_extractor.conv4",
        "conv4_1.0": "feature_extractor.conv4_1",
        "conv5.0": "feature_extractor.conv5",
        "conv5_1.0": "feature_extractor.conv5_1",
        "conv6.0": "feature_extractor.conv6",
    }

    # Transfer FlowNet encoder weights to DeepVO feature extractor
    for flownet_layer, deepvo_layer in mapping.items():
        if flownet_layer + ".weight" in flownet_state_dict:
            deepvo_state_dict[deepvo_layer + ".weight"] = flownet_state_dict[
                flownet_layer + ".weight"
            ]
        if flownet_layer + ".bias" in flownet_state_dict:
            deepvo_state_dict[deepvo_layer + ".bias"] = flownet_state_dict[
                flownet_layer + ".bias"
            ]

    # Load updated state_dict into DeepVO model
    deepvo_model.load_state_dict(deepvo_state_dict, strict=False)
    print("Successfully loaded FlowNet encoder weights into DeepVO.")

    return deepvo_model


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the given model.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
