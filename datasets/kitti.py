from pathlib import Path

import numpy as np
from PIL import Image
import torch
from typing import List, Tuple, Union
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from utils.data_utils import load_normalization


class KITTI(torch.utils.data.Dataset):
    """
    Dataloader for KITTI Visual Odometry Dataset
        http://www.cvlibs.net/datasets/kitti/eval_odometry.php

    Args:
        data_dpath (str): path to data sequences
        gt_dpath (str): path to poses
        camera_id (Union[str, int]): camera identifier (default="2")
        sequences (List[str]): list of sequence IDs to be loaded
        window_size (int): sliding window size for frames
        overlap (int): overlap size between consecutive windows
        normalize_gt (bool): flag to normalize the ground truth poses (default=False)
        transforms (callable): transformation to be applied on frames
    """

    def __init__(
        self,
        data_dpath: Union[Path, str] = "data/kitti",
        camera_id: Union[str, int] = "2",
        sequences: List[str] = ["00", "02", "08", "09"],
        window_size: int = 3,
        overlap: int = 1,
        normalize_gt: bool = False,
        transforms: transforms.Compose = None,
    ):
        self.data_dpath = Path(data_dpath)
        self.camera_id = str(camera_id)
        self.window_size = window_size
        self.overlap = overlap
        self.normalize_gt = normalize_gt
        self.transforms = transforms
        self.sequences = sequences

        # KITTI normalization parameters
        self.mean_angles, self.std_angles, self.mean_t, self.std_t = load_normalization(
            stats_file="datasets/dataset_stats.json", dataset_name="kitti"
        )

        # Build data dictionary with frames and ground truths for each sequence
        self.data_dict = self._build_data_dict(
            self.data_dpath, self.sequences, self.camera_id
        )

        # Build windowed dict based on the window size and overlap
        self.windowed_dict = self._create_windowed_dict(
            self.data_dict, self.window_size, self.overlap, self.sequences
        )

    def __len__(self) -> int:
        """Return the number of unique window indices."""
        return len(self.windowed_dict)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Retrieves the data for a specific window index.
        """
        # Get frames and ground truth of a given window index
        windowed_data = self.windowed_dict[idx]
        frames, gt_poses = windowed_data["frames"], windowed_data["ground_truth"]

        # Load and transform the images for the current window
        imgs = []
        for frame_fpath in frames:
            img = Image.open(frame_fpath).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            imgs.append(img)  # (C, H, W)

        # Stack images into a single tensor in channels dimention
        imgs = torch.cat(imgs, dim=0)  # (2*C, H, W)

        # Process the ground truth poses for the current window
        gt_poses = self._compute_relative_poses(gt_poses, self.normalize_gt)

        return imgs, gt_poses

    def _list_frames(self, frames_dpath: Path) -> List[str]:
        """
        Lists all the frame file paths in the given directory.

        Args:
            frames_dpath (Path): Path to the directory containing the frame files.

        Returns:
            List[str]: A list of sorted file paths for all frames.
        """
        # Return a sorted list of all PNG files in the directory
        return sorted([fpath for fpath in frames_dpath.glob("*.png")])

    def _load_ground_truth(self, gt_fpath) -> list:
        """
        Read ground truth poses in a given data path.

        Returns:
            list: List of ground truth pose vectors
        """
        gt = []
        with open(gt_fpath, "r") as f:
            lines = f.readlines()
        gt.extend([[float(value) for value in line.strip().split()] for line in lines])
        return gt

    def _build_data_dict(
        self, data_dpath: Path, sequences: list, camera_id: str
    ) -> dict:
        """
        Build a dictionary containing frame paths and corresponding ground truth data
        for each sequence in the dataset.

        This method iterates over all sequences provided, loads frame file paths and their
        associated ground truth data (translation and rotation). The data is stored in a dictionary
        where each key is a sequence name, and the value is another dictionary containing:
            - 'frames': A list of file paths to the frames.
            - 'ground_truth': A NumPy array of ground truth values for each frame.

        Args:
            data_dpath (Path): The base directory path where the sequence data is located.
            sequences (list): A list of sequence names (strings) to iterate over and load data for.
            camera_id (Union[str, int]): Camera identifier

        Returns:
            dict: A dictionary where each key is a sequence name, and the value
                is a dictionary containing 'frames' and 'ground_truth'.
        """
        data_dict = {}

        # Iterate over all sequences
        for sequence in sequences:
            sequence_dpath = data_dpath / "sequences" / sequence / f"image_{camera_id}"

            if sequence_dpath.is_dir():
                # Get all frame file paths
                frames_fpath = self._list_frames(sequence_dpath)

                # Load ground truth data (translation and rotation)
                gt_sequence = self._load_ground_truth(
                    data_dpath / "poses" / f"{sequence}.txt"
                )

                # Store the frames and ground truth in the data_dict
                data_dict[sequence] = {
                    "frames": frames_fpath,
                    "ground_truth": gt_sequence,
                }

        return data_dict

    def _create_windowed_dict(
        self, data_dict: dict, window_size: int, overlap: int, sequences: list
    ) -> dict:
        """
        Create a dictionary of windowed data for all sequences.

        This method generates windows of data from the provided data_dict. Each window contains
        a fixed number of frames (window_size) and is created with the specified overlap
        between windows.

        Args:
            data_dict (dict): A dictionary where each key is a sequence name
                and the value is another dictionary containing 'frames' and 'ground_truth'.
            window_size (int): The size of each window (number of frames).
            overlap (int): The number of frames that overlap between consecutive windows.
            sequences (list): A list of sequence names.

        Returns:
            dict: A dictionary where each key is a window index (w_idx)
                and the value is a dictionary containing 'frames' and 'ground_truth' for that window.
        """
        windowed_dict = {}
        w_idx = 0

        # Iterate through each unique sequence in the data_dict
        for sequence in sequences:
            if sequence in data_dict:
                sequence_frames = data_dict[sequence]["frames"]
                sequence_gt = data_dict[sequence]["ground_truth"]

                # Create windows for the current sequence
                row_idx = 0
                while row_idx + window_size <= len(sequence_frames):
                    # Extract windowed data
                    windowed_data = {
                        "frames": sequence_frames[row_idx : row_idx + window_size],
                        "ground_truth": sequence_gt[row_idx : row_idx + window_size],
                    }

                    # Store the windowed data in the windowed_dict
                    windowed_dict[w_idx] = windowed_data

                    # Update indices for the next window
                    row_idx += window_size - overlap
                    w_idx += 1

        return windowed_dict

    def _compute_relative_poses(
        self, poses: list, normalize_gt: bool = False
    ) -> np.ndarray:
        """
        Compute relative poses between consecutive frames in a window.

        Args:
            poses (list): List of poses in the window
            normalize_gt (bool): Normalize ground truth pose if True.

        Returns:
            np.ndarray: Flattened array of relative poses
        """
        y = []
        for i in range(1, len(poses)):
            pose_prev = np.vstack(
                [np.reshape(poses[i - 1], (3, 4)), [[0.0, 0.0, 0.0, 1.0]]]
            )
            pose_curr = np.vstack(
                [np.reshape(poses[i], (3, 4)), [[0.0, 0.0, 0.0, 1.0]]]
            )
            pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose_curr)

            # Rotation and translation
            t = pose_wrt_prev[:3, 3]
            rot = pose_wrt_prev[:3, :3]
            rot = R.from_matrix(rot)

            # Rotation to Euler angles
            angles = rot.as_euler("xyz")

            # Pose normalization
            if normalize_gt:
                angles, t = self._normalize_pose(angles, t)

            # Concatenate rotation and translation
            y.append(np.concatenate([angles, t]))

        return np.asarray(y)

    def _normalize_pose(
        self, angles: List[float], t: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes the angles and translations using mean and standard
        deviation values.

        Args:
            angles (List[float]): List of angle values to be normalized.
            t (List[float]): List of translation values to be normalized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing normalized angles and translations.
        """
        normalized_angles = (np.array(angles) - self.mean_angles) / self.std_angles
        normalized_t = (np.array(t) - self.mean_t) / self.std_t
        return normalized_angles, normalized_t


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dpath = "C:/workspace/data/kitti"
    sequences = ["00", "01"]

    def visualize_samples(dataset, num_samples=5):
        """
        Visualizes a few image samples and plots their 2D and 3D trajectory.
        """
        # Get the first window of data
        imgs, _ = dataset[0]  # Get the first window
        imgs = torch.split(imgs, 3, dim=0)

        # Ensure we don't try to visualize more samples than available
        num_frames = len(imgs)  # Number of frames in the window
        num_samples = min(
            num_samples, num_frames
        )  # Update num_samples to available count

        # Visualize image samples
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        for i in range(num_samples):
            img = (
                imgs[i].numpy().transpose(1, 2, 0)
            )  # Change shape from (C, H, W) to (H, W, C)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Sample {i+1}")

        plt.show()
        plt.close()

    def visualize_sequence_trajectory(gt_poses: np.ndarray, sequence: str) -> None:
        """
        Visualizes athe ground truth trajectory of a sequence.
        """
        # Get translation from ground truth
        t = gt_poses[:, [3, 7, 11]]

        # Plot 2D trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(t[:, 0], t[:, 2])
        plt.title("2D Trajectory")
        plt.xlabel("X Position")
        plt.ylabel(
            "Z Position"
        )  # Correcting Y label to Z, assuming the third column is Z
        plt.grid()
        plt.title(sequence)
        plt.show()

    # Create dataloader
    preprocess = transforms.Compose(
        [
            transforms.Resize((384, 1280)),
            transforms.ToTensor(),
        ]
    )
    dataset = KITTI(
        data_dpath=data_dpath,
        sequences=sequences,
        window_size=2,
        overlap=1,
        transforms=preprocess,
    )

    imgs, gt = dataset[0]  # Get the first window
    print("Images shape:", imgs.shape)
    print("Ground truth shape:", gt.shape)

    visualize_samples(dataset)
    for sequence in sequences:
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        visualize_sequence_trajectory(np.asarray(gt_poses), sequence)
