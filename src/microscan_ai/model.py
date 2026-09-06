from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small baseline CNN with resolution-independent pooling."""

    def __init__(self, input_channels: int, num_classes: int, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def build_model(architecture: str, input_channels: int, num_classes: int, width: int) -> nn.Module:
    if architecture != "tiny_cnn":
        raise ValueError(f"Unsupported architecture: {architecture}")
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    return TinyCNN(input_channels, num_classes, width)
