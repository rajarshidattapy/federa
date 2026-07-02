"""Mutable server-side state for one federated training run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from federa.models.base import FederatedModel
from federa.training.fedavg import ClientUpdate
from federa.utils.metrics import MetricsTracker


@dataclass(slots=True)
class GlobalState:
    model: FederatedModel
    round_number: int = 0
    total_rounds: int = 10
    pending_updates: list[ClientUpdate] = field(default_factory=list)
    metrics: MetricsTracker = field(default_factory=MetricsTracker)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def add_update(self, update: ClientUpdate) -> None:
        self.pending_updates.append(update)

    def clear_updates(self) -> None:
        self.pending_updates.clear()

    def is_training_complete(self) -> bool:
        return self.round_number >= self.total_rounds
