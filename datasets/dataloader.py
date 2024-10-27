from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import random_split, DataLoader
from datasets.transforms import get_transforms

from datasets.kitti import KITTI


def build_dataloader(
    data_params: dict,
    split: str,
    sequence: Optional[str] = None,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Builds and returns the DataLoader(s) for the specified dataset and split.

    Args:
        data_params (dict): Dictionary containing the dataset configuration and parameters.
        split (str): The dataset split, either 'train', 'val', or 'test'. Defaults to 'train'.
        sequence (Optional[str]): Sequence ID to load specific sequences. Defaults to None.

    Returns:
        Union[Tuple[DataLoader, DataLoader], DataLoader]:
        A tuple of (train_loader, val_loader) for 'train', or a test_loader for 'test'.
    """
    # Extract parameters
    dataset_name = data_params["dataset"]
    data_dpath = data_params["data_dpath"]
    image_size = data_params["image_size"]
    window_size = data_params["window_size"]
    overlap = data_params["overlap"]
    batch_size = data_params["bsize"]
    num_workers = data_params["num_workers"]
    normalize_gt = data_params.get("normalize_gt", False)

    # Get transforms based on the dataset
    transforms = get_transforms(dataset_name, image_size)

    if dataset_name == "kitti":
        # Define sequences to create dataset
        if sequence:
            sequences = [sequence]
        else:
            if split == "train":
                sequences = data_params["training_sequences"]
            else:
                sequences = data_params["test_sequences"]
        dataset = KITTI(
            data_dpath=data_dpath,
            sequences=sequences,
            window_size=window_size,
            overlap=overlap,
            transforms=transforms,
            normalize_gt=normalize_gt,
        )

    else:
        raise ValueError(f"--- Undefined dataset: {dataset_name} ---")

    # Create validation data for 'train' split
    if split == "train":
        val_split = data_params.get("val_split", 0.1)
        nb_samples_val = round(val_split * len(dataset))

        # Set the random seed for reproducibility
        generator = torch.Generator().manual_seed(2)
        train_data, val_data = random_split(
            dataset,
            [len(dataset) - nb_samples_val, nb_samples_val],
            generator=generator,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    elif split == "test":
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return test_loader
