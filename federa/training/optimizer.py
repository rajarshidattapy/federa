"""Optimizer factory so config (not code) selects the local training optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch

OptimizerName = Literal["sgd", "adam", "adamw"]


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    name: OptimizerName = "sgd",
    lr: float = 0.01,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    params = list(parameters)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")
