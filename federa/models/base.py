"""Abstract contract every model wrapper in Federa must satisfy.

The TypeScript prototype hard-coded a single feed-forward network
(matrix.ts + activations.ts + network.ts). Federa instead standardizes on
this narrow interface so *any* torch.nn.Module -- a CNN, a transformer, a
LoRA-adapted LLM -- can be plugged into the same FedAvg/FedProx machinery
without the framework knowing anything about its architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class FederatedModel(ABC):
    """Adapter interface between a trainable model and the federated engine."""

    @abstractmethod
    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU-agnostic copy of the model's parameters/buffers."""

    @abstractmethod
    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load a state dict produced by `get_weights` (e.g. after aggregation)."""

    @abstractmethod
    def parameters(self) -> list[torch.nn.Parameter]:
        """Trainable parameters, for constructing an optimizer."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run the wrapped model's forward pass."""

    @abstractmethod
    def train_mode(self) -> None: ...

    @abstractmethod
    def eval_mode(self) -> None: ...

    @abstractmethod
    def to(self, device: torch.device | str) -> FederatedModel: ...
