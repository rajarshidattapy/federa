"""The client-side node: local training, scheduling, and the coordinator connection."""

from federa.client.node import SwarmNode
from federa.client.scheduler import TrainingScheduler
from federa.client.trainer import LocalTrainer, LocalTrainingResult
from federa.client.websocket import ClientConnection

__all__ = [
    "ClientConnection",
    "LocalTrainer",
    "LocalTrainingResult",
    "SwarmNode",
    "TrainingScheduler",
]
