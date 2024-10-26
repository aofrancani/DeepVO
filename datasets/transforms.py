from typing import Tuple
from torchvision import transforms


def get_transforms(dataset: str, image_size: Tuple[int, int]):
    """Returns the appropriate preprocessing pipeline for the specified dataset."""

    if dataset == "kitti":
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.34721234, 0.36705238, 0.36066107],
                    std=[0.30737526, 0.31515116, 0.32020183],
                ),
            ]
        )
    else:
        raise ValueError(f"Transforms not defined for '{dataset}'")
