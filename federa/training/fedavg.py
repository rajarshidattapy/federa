"""Federated Averaging (FedAvg).

Successor to `Coordinator.aggregate` in the TypeScript prototype's
coordinator.ts. The TS version averaged client matrices with an unweighted
mean; this implementation follows the original FedAvg algorithm (McMahan et
al., 2017, Algorithm 1) and weights each client's contribution by how many
local samples it trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ClientUpdate:
    client_id: str
    num_samples: int
    weights: dict[str, torch.Tensor]


def federated_average(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Sample-weighted average: w_{t+1} = sum_k (n_k / n) * w_{t+1}^k."""
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates")

    total_samples = sum(update.num_samples for update in updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across updates must be positive")

    reference_keys = updates[0].weights.keys()
    for update in updates[1:]:
        if update.weights.keys() != reference_keys:
            raise ValueError("All client updates must share the same parameter keys")

    averaged: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        reference = updates[0].weights[key]
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for update in updates:
                client_weight = update.num_samples / total_samples
                accumulator += update.weights[key].to(torch.float32) * client_weight
            averaged[key] = accumulator.to(reference.dtype)
        else:
            # Non-float buffers (e.g. BatchNorm's num_batches_tracked) aren't
            # meaningfully averaged -- keep the most recent client's value.
            averaged[key] = updates[-1].weights[key].clone()

    return averaged
