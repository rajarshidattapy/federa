"""Glue between raw client updates and a training-strategy-specific aggregate."""

from __future__ import annotations

import torch

from federa.training.fedavg import ClientUpdate, federated_average
from federa.utils.config import CoordinatorSettings


class Aggregator:
    """Runs the configured aggregation strategy for a completed round.

    FedProx's proximal term (see `federa.training.fedprox`) is applied
    client-side during local training; aggregation itself is the same
    sample-weighted average as FedAvg for both strategies.
    """

    def __init__(self, settings: CoordinatorSettings) -> None:
        self.settings = settings

    def aggregate(self, updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
        if self.settings.aggregation_strategy in ("fedavg", "fedprox"):
            return federated_average(updates)
        raise ValueError(f"Unknown aggregation strategy: {self.settings.aggregation_strategy}")
