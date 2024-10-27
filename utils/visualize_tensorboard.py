import argparse
from pathlib import Path
import subprocess
import sys

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
    train_loss = [scalar.value for scalar in event_acc.Scalars("train_loss")]
    val_loss = [scalar.value for scalar in event_acc.Scalars("val_loss")]

    # Get epochs
    epochs = [scalar.step for scalar in event_acc.Scalars("train_loss")]

    return epochs, train_loss, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard to visualize logs."
    )
    parser.add_argument(
        "--log_dir", type=str, help="Directory with TensorBoard log files."
    )
    args = parser.parse_args()

    # Launch TensorBoard
    launch_tensorboard(log_dir=args.log_dir)
