"""Concrete `FederatedModel` backed by an arbitrary `torch.nn.Module`."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from federa.models.base import FederatedModel


class TorchModelAdapter(FederatedModel):
    """Wraps any `nn.Module` so it can be trained, serialized, and averaged.

    This is the direct replacement for the TypeScript `NeuralNetwork` class:
    instead of hand-rolled matrix math and a fixed 3-layer topology, weight
    extraction/loading is delegated to PyTorch's `state_dict` machinery,
    which works for any architecture.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone().cpu() for k, v in self.module.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        device = next(self.module.parameters(), torch.empty(0)).device
        self.module.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=True)

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.module.parameters())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def train_mode(self) -> None:
        self.module.train()

    def eval_mode(self) -> None:
        self.module.eval()

    def to(self, device: torch.device | str) -> TorchModelAdapter:
        self.module.to(device)
        return self


def wrap_model(module: nn.Module) -> TorchModelAdapter:
    """Convenience constructor: `wrap_model(MyModel())`."""
    return TorchModelAdapter(module)
