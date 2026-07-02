"""Starts the Federa coordinator for the MNIST FedAvg example.

All tuning (rounds, min_clients_per_round, privacy, quantization, ...) comes
from `federa.utils.config.CoordinatorSettings` via environment variables --
see examples/README.md. Sensible defaults (min_clients_per_round=2,
rounds=10) apply if none are set.

Run: python -m examples.mnist_fedavg.server
"""

from __future__ import annotations

from examples.mnist_fedavg.model import build_model
from federa import Coordinator
from federa.models.pytorch import wrap_model


def main() -> None:
    Coordinator(wrap_model(build_model())).run()


if __name__ == "__main__":
    main()
