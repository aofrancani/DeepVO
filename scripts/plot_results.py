import queue
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets.kitti import KITTI
from utils.data_utils import load_normalization
from utils.config_utils import load_config
from scipy.spatial.transform import Rotation as R


def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file
    Args:
        poses (ndarray): list with all 4x4 pose matrix
        sequence (str): sequence of KITTI dataset
        save_dir (str): path to save pose
    """
    save_dir = Path(save_dir)

    # create directory
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    output_filename = save_dir / f"{sequence}.txt"
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def recover_trajectory_and_poses(poses, normalize_gt):

    predicted_poses = []
    # recover predicted trajectory
    predicted_trajectory = []
    for i in range(len(poses) - 1):
        if i == 0:
            T = np.eye(4)

        angles = poses[i, :3]
        t = poses[i, 3:]

        # Undo normalization
        if normalize_gt:
            mean_angles, std_angles, mean_t, std_t = load_normalization(
                stats_file="datasets/dataset_stats.json", dataset_name="kitti"
            )
            [x, y, z] = np.multiply(angles, std_angles) + mean_angles
            t = np.multiply(t, std_t) + mean_t

        # Euler angles to rotation matrix
        rot = R.from_euler("xyz", angles)
        rot = rot.as_matrix()

        T_r = np.concatenate(
            (
                np.concatenate([rot, np.reshape(t, (3, 1))], axis=1),
                [[0.0, 0.0, 0.0, 1.0]],
            ),
            axis=0,
        )
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m testing.plot_results_vbr <config_fpath> <checkpoint_fname>"
        )
        sys.exit(1)

    config_fpath = sys.argv[1]
    checkpoint_name = sys.argv[2]

    # Load hyperparameters
    config = load_config(config_fpath)
    checkpoint_fpath = Path(config["checkpoint"]["checkpoint_dpath"]) / checkpoint_name

    # Extract configuration parameters
    data_params = config.get("data", {})

    for sequence in data_params["test_sequences"]:
        # read ground test data and predicted poses
        pred_path = (
            checkpoint_fpath / data_params["dataset"] / f"pred_poses_{sequence}.npy"
        )
        try:
            pred_poses = np.load(pred_path)
        except:
            continue

        # post processing and recover trajectory
        poses = pred_poses.squeeze(1)
        poses = np.asarray(pred_poses)
        pred_poses, pred_trajectory = recover_trajectory_and_poses(
            poses, normalize_gt=data_params.get("normalize_gt", False)
        )

        save_trajectory(
            pred_poses,
            sequence,
            save_dir=checkpoint_fpath / data_params["dataset"] / "pred_poses",
        )

        # get ground truth trajectories
        dataset = KITTI(
            data_dpath=data_params["data_dpath"],
            sequences=[sequence],
        )
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        gt_trajectory = np.asarray(gt_poses)[:, [3, 7, 11]]

        plt.figure()
        pred_trajectory = np.asarray(pred_trajectory)
        plt.plot([t[0] for t in pred_trajectory], [t[2] for t in pred_trajectory], "b")
        plt.plot([t[0] for t in gt_trajectory], [t[2] for t in gt_trajectory], "r")
        plt.grid()
        plt.title(f"{sequence}")
        plt.xlabel("Translation in x direction [m]")
        plt.ylabel("Translation in z direction [m]")
        plt.legend(["estimated", "ground truth"])

        # create checkpoints folder
        save_dir = checkpoint_fpath / data_params["dataset"] / "plots"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"pred_traj_{sequence}.png")
