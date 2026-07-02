"""A small CNN used by the MNIST FedAvg example."""

from __future__ import annotations

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model() -> MnistCNN:
    return MnistCNN()


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((predictions.argmax(dim=1) == targets).float().mean().item())
