###########################################################################
## This is an unofficial PyTorch implementation of the DeepVO model.
## [source]: https://senwang.gitlab.io/DeepVO/files/wang2017DeepVO.pdf
## Author: André Françani, 2024
###########################################################################

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    """
    CNN backbone to extract spatial features from input images.
    This module takes in a single image and outputs a 1D feature vector per image.
    """

    def __init__(
        self,
        input_channels: int = 6,
        input_res: Tuple[int, int] = (384, 1280),
        conv_dropout=0.1,
    ) -> None:
        super(CNN, self).__init__()

        # CNN layers
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=conv_dropout)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # Get features dimention (1024 * (W/64) * (H/64) = (W * H) / 4)
        x = torch.zeros(1, input_channels, *input_res)
        self.features_dim = int(np.prod(self.forward(x).size()))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the CNN feature extractor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Extracted features of shape (batch_size, features_dim).
        """
        # Pass through each layer
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.dropout(self.relu(self.conv3_1(x)))
        x = self.dropout(self.relu(self.conv4(x)))
        x = self.dropout(self.relu(self.conv4_1(x)))
        x = self.dropout(self.relu(self.conv5(x)))
        x = self.dropout(self.relu(self.conv5_1(x)))
        x = self.dropout(self.conv6(x))
        x = self.flatten(x)
        return x


class DeepVO(nn.Module):
    """DeepVO model for monocular visual odometry.

    This model estimates relative pose (translation and rotation) between frames using
    a CNN for feature extraction and an LSTM for temporal dependency modeling.
    """

    def __init__(
        self,
        input_channels: int = 6,
        input_res: list = [192, 640],
        hidden_size: int = 1000,
        lstm_layers: int = 2,
        output_size: int = 6,
        lstm_dropout: float = 0.2,
        conv_dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_channels (int): Number of channels in the input image. Default is 3 for RGB images.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            lstm_layers (int): Number of layers in the LSTM.
            output_size (int): Dimensionality of the output pose (6 for 3D translation and rotation).
            lstm_dropout (float): Dropout in LSTM layer.
        """
        super(DeepVO, self).__init__()

        self.feature_extractor = CNN(
            input_channels=input_channels,
            input_res=input_res,
            conv_dropout=conv_dropout,
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_extractor.features_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: Tensor, hidden_state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        """Forward pass through the DeepVO model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).
            hidden_state (Optional[Tuple[Tensor, Tensor]]): Optional initial hidden state and cell state for the LSTM.

        Returns:
            Tensor: Output poses.
        """
        batch_size = x.size(0)

        # Features extractor with CNN
        features = self.feature_extractor(x)

        # Reshape features for the LSTM input (batch_size x 1 x features_dim)
        features = features.view(batch_size, 1, -1)

        # Forward pass through LSTM
        lstm_out, hidden_state = self.lstm(features, hidden_state)

        # Forward pass through fully connected layer to predict pose
        y = self.fc(lstm_out)

        return y

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Initialize hidden state for the LSTM with zeros.

        Args:
            batch_size (int): The batch size for initializing the hidden state.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - Hidden state of shape (num_layers, batch_size, hidden_size).
                - Cell state of shape (num_layers, batch_size, hidden_size).
        """
        weight = next(self.parameters()).data
        hidden_state = (
            weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
            weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
        )
        return hidden_state


def count_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in the model.

    Args:
        model (nn.Module): The model instance to count parameters for.

    Returns:
        int: Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from pprint import pprint as pp
    from models.build_model import load_pretrained_flownet

    batch_size = 8
    input_channels = 6
    input_res = (384, 1280)

    # Build the model
    model = DeepVO(
        input_channels=input_channels,
        input_res=input_res,
        hidden_size=1000,
        lstm_layers=2,
        output_size=6,
    )
    print(f"# Parameters: {count_parameters(model):,}")

    # Dummy input
    x = torch.randn(
        batch_size, input_channels, input_res[0], input_res[1]
    )  # (B x C x H x W)

    # Initialize hidden state for the LSTM
    hidden_state = model.init_hidden(batch_size=batch_size)

    # Inference
    with torch.no_grad():
        y = model(x, hidden_state)
    print(f"y.shape: {y.shape}")  # (batch_size, 1, output_size)

    # load pretrained FlowNet
    model = load_pretrained_flownet(model)
