from typing import Tuple

import json
import numpy as np

from scipy.spatial.transform import Rotation as R


def rotation_to_euler(
    rotation: np.ndarray, seq: str = "xyz"
) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.
        seq (str): Order of rotations for the Euler angles (default 'xyz').

    Returns:
        Tuple[float, float, float]: Euler angles in radians.
    """
    rotation = R.from_matrix(rotation)
    return rotation.as_euler(seq, degrees=False)


def euler_to_rotation(
    euler_angles: Tuple[float, float, float], seq: str = "xyz"
) -> np.ndarray:
    """Convert Euler angles to a rotation matrix.

    Args:
        euler_angles (Tuple[float, float, float]): Euler angles in radians.
        seq (str): Order of rotations for the Euler angles (default 'xyz').

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rotation = R.from_euler(seq, euler_angles, degrees=False)
    return rotation.as_matrix()


def quaternion_to_rotation_matrix(q):
    # q is expected to be in the format [qx, qy, qz, qw]
    r = R.from_quat(q)  # Convert to Rotation object
    return r.as_matrix()  # Get the rotation matrix


def load_normalization(stats_file: str, dataset_name: str):
    """
    Loads normalization parameters from a JSON file for the specified dataset.

    Args:
        stats_file (str): Path to the JSON file.
        dataset_name (str): Name of the dataset to retrieve parameters from (e.g., "kitti", "7scenes").

    Returns:
        mean_angles (np.array): Mean angles for the dataset.
        std_angles (np.array): Standard deviation of angles for the dataset.
        mean_t (np.array): Mean translation for the dataset.
        std_t (np.array): Standard deviation of translation for the dataset.
    """
    # Read the JSON file
    with open(stats_file, "r") as f:
        data = json.load(f)

    # Ensure the dataset_name exists in the JSON data
    if dataset_name not in data:
        raise ValueError(f"Dataset '{dataset_name}' not found in the JSON file.")

    # Extract normalization parameters
    dataset_params = data[dataset_name]
    mean_angles = np.array(dataset_params["mean_angles"])
    std_angles = np.array(dataset_params["std_angles"])
    mean_t = np.array(dataset_params["mean_t"])
    std_t = np.array(dataset_params["std_t"])

    return mean_angles, std_angles, mean_t, std_t
