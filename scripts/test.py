from pathlib import Path
from typing import Union

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from utils.config_utils import load_config
from utils.checkpoint_utils import load_checkpoint
from datasets.dataloader import build_dataloader
from models.build_model import build_model, load_model_state


class PosePredictor:
    """
    Class for predicting poses using a deep learning-based model.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
    ):
        """
        Initializes the PosePredictor with the provided model and dataloader.

        Args:
            model (nn.Module): The model to use for pose prediction.
            dataloader (torch.utils.data.DataLoader): The DataLoader for loading data.
            device (Union[torch.device, str]): Either a CPU or a CUDA device.
        """
        self.device = device
        self.model = model.to(self.device)
        self.model = self.model.eval()
        self.dataloader = dataloader
        self.window_size = dataloader.dataset.window_size

    def predict(self) -> np.ndarray:
        """Runs the pose prediction over the dataset.

        Returns:
            np.ndarray: Array of predicted poses.
        """
        pred_poses = torch.zeros((1, self.window_size - 1, 6), device=self.device)

        with tqdm(self.dataloader, unit="batch") as batchs:
            for images, gt in batchs:
                images, gt = images.to(self.device), gt.to(self.device)

                with torch.no_grad():
                    pred_pose = self.model(images.float())
                pred_pose = torch.reshape(
                    pred_pose, (self.window_size - 1, 6)
                ).unsqueeze(dim=0)
                pred_poses = torch.cat((pred_poses, pred_pose), dim=0)

        # Return poses as a numpy array
        return pred_poses.cpu().detach().numpy()
        # return pred_poses[1:, :, :].cpu().detach().numpy()


def save_predictions(
    predictions: np.ndarray, save_dpath: Union[str, Path], sequence: str
) -> None:
    """Saves the predicted poses to a NumPy file.

    Args:
        predictions (np.ndarray): Predicted poses to be saved.
        save_dpath (Union[str, Path]): Directory path to save predictions
        sequence (str): Name of the sequence
    """

    save_dpath = Path(save_dpath)

    # Create directory if it does not exist
    save_dpath.mkdir(parents=True, exist_ok=True)

    np.save(
        save_dpath / f"pred_poses_{sequence}.npy",
        predictions,
    )


def main(config_fpath, checkpoint_name):

    # Load hyperparameters
    config = load_config(config_fpath)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load checkpoint
    checkpoint_params = config.get("checkpoint", {})
    checkpoint_params["checkpoint_name"] = checkpoint_name
    checkpoint_fpath = (
        Path(checkpoint_params["checkpoint_dpath"])
        / checkpoint_params["checkpoint_name"]
    )
    checkpoint = load_checkpoint(checkpoint_fpath)

    # Build the model
    print("Building model...")
    model = build_model(config.get("model", {}), device)
    model = load_model_state(model, checkpoint)

    # Predict for each test sequence
    for sequence in config["data"]["test_sequences"]:
        print(f"Sequence: {sequence}")

        # Build dataloader
        dataloader = build_dataloader(
            config.get("data", {}), split="test", sequence=sequence
        )

        # Create PosePredictor instance
        predictor = PosePredictor(model, dataloader, device)

        # Perform predictions
        predicted_poses = predictor.predict()

        # Save predictions
        save_dpath = checkpoint_fpath / config["data"]["dataset"]
        save_predictions(predicted_poses, save_dpath, sequence)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m testing.inference_vbr <config_fpath> <checkpoint_fname>"
        )
        sys.exit(1)

    config_fpath = sys.argv[1]
    checkpoint_name = sys.argv[2]

    # Run main processing
    main(config_fpath, checkpoint_name)
