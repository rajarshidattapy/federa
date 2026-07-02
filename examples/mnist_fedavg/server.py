"""Starts the Federa coordinator for the MNIST FedAvg example.

Run: python -m examples.mnist_fedavg.server
"""

from __future__ import annotations

from federa import Coordinator
from federa.models.pytorch import wrap_model
from federa.utils.config import CoordinatorSettings

from examples.mnist_fedavg.model import build_model


def main() -> None:
    settings = CoordinatorSettings(min_clients_per_round=2, rounds=5)
    Coordinator(wrap_model(build_model()), settings=settings).run()


if __name__ == "__main__":
    main()
