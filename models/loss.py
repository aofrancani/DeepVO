import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    Computes a weighted MSE loss for angle and translation components.

    Args:
        window_size (int): Used to reshape tensors for weighted loss calculation.
        alpha (float, optional): Weight for the angle loss component.
    """

    def __init__(
        self,
        window_size: int,
        alpha: float,
    ):
        super(WeightedMSELoss, self).__init__()
        self.window_size = window_size
        self.alpha = float(alpha)
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        gt_angles = y_true[:, :, :3].flatten()
        gt_translation = y_true[:, :, 3:].flatten()

        # Separate angles and translation for predicted
        y_pred = torch.reshape(y_pred, (y_pred.shape[0], self.window_size - 1, 6))
        estimated_angles = y_pred[:, :, :3].flatten()
        estimated_translation = y_pred[:, :, 3:].flatten()

        # Calculate weighted losses for angles and translation
        loss_angles = self.mse_loss(estimated_angles, gt_angles.float())
        loss_translation = self.mse_loss(estimated_translation, gt_translation.float())

        # Compute final weighted loss
        loss = loss_translation + self.alpha * loss_angles

        return loss


def get_criterion(
    loss_name: str,
    window_size: int = 3,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Instantiates and returns a loss function based on the provided loss name.

    Args:
        loss_name (str): The name of the loss function. Options are 'mse' for standard MSE loss
                         and 'weighted_mse' for the custom weighted MSE loss.
        window_size (int, optional): Used to reshape tensors in the 'weighted_mse' loss.
        alpha (float, optional): Weight for the angle loss in 'w_mse'.

    Returns:
        nn.Module: An instance of the specified loss function.
    """
    if loss_name.lower() == "mse":
        return nn.MSELoss()  # Unweighted MSE loss

    elif loss_name.lower() == "w_mse":
        return WeightedMSELoss(window_size=window_size, alpha=alpha)

    else:
        raise ValueError(
            f"Unsupported loss name: {loss_name}. Available options: 'mse', 'weighted_mse'."
        )


if __name__ == "__main__":
    # Define dummy inputs for testing
    batch_size = 4
    window_size = 3

    # Dummy ground truth and predictions
    y_true = torch.rand(batch_size, (window_size - 1) * 6)
    y_pred = torch.rand(batch_size, (window_size - 1) * 6)

    # Test MSELoss
    mse_loss_fn = get_criterion("mse")
    mse_loss = mse_loss_fn(y_pred, y_true)
    print("MSE Loss:", mse_loss.item())

    # Test WeightedMSELoss with a weight for angle loss
    weighted_loss_fn = get_criterion("w_mse", window_size=window_size, alpha=2)
    weighted_loss = weighted_loss_fn(y_pred, y_true)
    print("Weighted MSE Loss:", weighted_loss.item())
