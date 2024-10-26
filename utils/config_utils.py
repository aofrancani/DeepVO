from typing import Dict
from pathlib import Path

import json


def load_config(config_fpath: str) -> Dict:
    """
    Loads configuration from a JSON file.

    Args:
        config_fpath (str): Path to the JSON configuration file.

    Returns:
        Dict: Configuration parameters.
    """
    with open(config_fpath, "r") as file:
        config = json.load(file)

    # Extracts config name
    config_name = get_experiment_name(config_fpath)

    # Extend config file with known relations and deterministic values
    config = extend_config(config, config_name)

    return config


def extend_config(config: Dict, config_name: str) -> Dict:
    """
    Extends the given configuration dictionary by adding new keys based on
    deterministic calculations and relations from existing fields.

    Args:
        config (Dict): The original configuration dictionary.
        config_name (str): Name of the config file.

    Returns:
        Dict: The extended configuration dictionary with new keys and values.

    Raises:
        KeyError: If any expected key is missing from the config.
        ValueError: If a field value is invalid for calculations.
    """

    # Check whether all the required fields are present
    check_required_keys(config)

    # Define new model parameters
    window_size = config["data"]["window_size"]
    config["model"]["image_size"] = config["data"]["image_size"]
    config["model"]["input_channels"] = window_size * 3
    config["model"]["output_size"] = 6  # 6-DoF
    config["model"]["num_frames"] = window_size

    # Update checkpoint directory path with exp name
    config["checkpoint"]["checkpoint_dpath"] = (
        Path(config["checkpoint"]["checkpoint_dpath"]) / config_name
    )

    return config


def check_required_keys(config: Dict) -> None:
    """
    Checks if all the required keys, including nested keys, are present in the configuration dictionary.

    Args:
        config (Dict): The configuration dictionary.

    Raises:
        KeyError: If any required key is missing from the config.
    """
    required_keys = [
        "data.window_size",
        "data.image_size",
        "checkpoint.checkpoint_dpath",
    ]
    for key in required_keys:
        keys = key.split(".")  # Split the key by '.' to access nested keys
        current_dict = config
        for sub_key in keys:
            if sub_key not in current_dict:
                raise KeyError(f"Key '{key}' is required in the configuration file.")
            current_dict = current_dict[sub_key]  # Drill down to the next level


def get_experiment_name(config_fpath: str) -> str:
    """
    Extracts the experiment name from a given configuration file path.

    The experiment name is defined as the part of the file name without the
    extension. For example, if the input is "configs/exp1.json", the function
    will return "exp1".

    Args:
        config_fpath (str): The path to the configuration file. It is expected
                            to be a string in the format 'path/exp_name.extension'.

    Returns:
        str: The experiment name extracted from the file path.
    """
    # Get config file name
    config_fname = config_fpath.split("/")[-1]

    # Remove .json extension
    exp_name = config_fname.split(".")[0]

    return exp_name


if __name__ == "__main__":
    from pprint import pprint as pp

    config_fpath = "configs/exp1.json"
    config = load_config(config_fpath)
    pp(config)
