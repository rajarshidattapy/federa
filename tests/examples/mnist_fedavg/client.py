"""Runs one Federa SwarmNode client for the MNIST FedAvg example.

Partitions MNIST by index modulo `--num-clients` so each client trains on a
disjoint shard, then joins the coordinator and trains until it signals the
run is complete.

Requires torchvision for the MNIST dataset: `pip install torchvision`.
Run: python -m examples.mnist_fedavg.client --client-id 0 --num-clients 3
"""

from __future__ import annotations

import argparse

import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

from examples.mnist_fedavg.model import accuracy, build_model
from federa import SwarmNode


def _load_partition(client_id: int, num_clients: int) -> Subset:
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    indices = list(range(client_id, len(full_dataset), num_clients))
    return Subset(full_dataset, indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://localhost:8000")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()

    node = SwarmNode(
        server=args.server,
        model=build_model(),
        dataset=_load_partition(args.client_id, args.num_clients),
        client_id=f"mnist-client-{args.client_id}",
        loss_fn=nn.CrossEntropyLoss(),
        metric_fn=accuracy,
    )
    node.start_training()


if __name__ == "__main__":
    main()
