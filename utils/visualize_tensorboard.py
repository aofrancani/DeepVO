import argparse
from pathlib import Path
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def launch_tensorboard(log_dir: str) -> None:
    """Launches TensorBoard to visualize the logs.

    Args:
        log_dir (str): Path to the directory containing TensorBoard event files.
    """

    log_dir = Path(log_dir)

    # Check if the log directory exists
    if not log_dir.is_dir():
        print(f"Error: Log directory '{str(log_dir)}' does not exist.")
        sys.exit(1)

    # Run TensorBoard
    subprocess.run(["tensorboard", "--logdir", log_dir])


def extract_losses_from_tensorboard(log_dir: str):
    # Load the TensorBoard event file
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Extract train and val losses
    train_loss = [scalar.value for scalar in event_acc.Scalars("loss/train")]
    val_loss = [scalar.value for scalar in event_acc.Scalars("loss/val")]

    # Get epochs
    epochs = [epoch + 1 for epoch in range(len(train_loss))]

    return epochs, train_loss, val_loss


def plot_training_loss(
    train_loss: list, val_loss: list, epochs: list, save_dpath: str = "loss"
) -> None:
    """
    Plots training and validation loss over epochs.

    Args:
        train_loss (List[float]): List of training loss values.
        val_loss (List[float]): List of validation loss values.
        epochs (List[int]): List of epoch indices corresponding to loss values.
        save_dpath (str): File path (without extension) to save the plot (default is "loss").
    """
    # Set LaTeX-like formatting
    rcParams.update({"font.family": "serif", "font.size": 16})
    # Create figure and plot data
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, color="k", linewidth=2, label="Training")
    plt.plot(epochs, val_loss, color="r", linewidth=2, label="Validation")
    plt.grid()
    plt.legend(fontsize=20, loc="upper right")
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.tight_layout()

    # Save plot
    plt.savefig(
        f"{save_dpath}/learning_curve.png", bbox_inches="tight", pad_inches=0, dpi=100
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard to visualize logs."
    )
    parser.add_argument(
        "--log_dir", type=str, help="Directory with TensorBoard log files."
    )
    args = parser.parse_args()

    # make plot
    epochs, train_loss, val_loss = extract_losses_from_tensorboard(log_dir=args.log_dir)
    plot_training_loss(train_loss, val_loss, epochs, args.log_dir)

    # Launch TensorBoard
    launch_tensorboard(log_dir=args.log_dir)
