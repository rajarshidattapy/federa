"""Federated aggregation algorithms, checkpointing, and optimizer construction."""

from federa.training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from federa.training.fedavg import ClientUpdate, federated_average
from federa.training.fedprox import fedprox_loss, fedprox_proximal_term
from federa.training.optimizer import build_optimizer

__all__ = [
    "ClientUpdate",
    "build_optimizer",
    "federated_average",
    "fedprox_loss",
    "fedprox_proximal_term",
    "load_checkpoint",
    "latest_checkpoint",
    "save_checkpoint",
]
