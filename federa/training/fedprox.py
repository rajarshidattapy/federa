"""FedProx: FedAvg with a proximal term for non-IID client data.

Adds `(mu / 2) * ||w_local - w_global||^2` to the local loss so a client
with unusual local data can't drag its update too far from the global
model in a single round (Li et al., 2020, "Federated Optimization in
Heterogeneous Networks").
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def fedprox_proximal_term(
    local_params: Iterable[torch.nn.Parameter],
    global_weights: Iterable[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    local_list = list(local_params)
    global_list = list(global_weights)

    if mu == 0 or not local_list:
        return torch.zeros(())

    term = torch.zeros((), device=local_list[0].device)
    for local, global_param in zip(local_list, global_list, strict=True):
        term = term + torch.sum((local - global_param.to(local.device)) ** 2)
    return (mu / 2.0) * term


def fedprox_loss(
    base_loss: torch.Tensor,
    local_params: Iterable[torch.nn.Parameter],
    global_weights: Iterable[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    return base_loss + fedprox_proximal_term(local_params, global_weights, mu)
