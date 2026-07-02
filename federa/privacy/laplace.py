"""Laplace-mechanism differential privacy.

Direct successor to `DifferentialPrivacy.applyNoise` in the TypeScript
prototype's privacy.ts, generalized from the bespoke `Matrix` type to
arbitrary tensors and full model state dicts.
"""

from __future__ import annotations

import torch


class LaplaceMechanism:
    """Adds Laplace(0, sensitivity/epsilon) noise to protect individual updates.

    Under epsilon-differential privacy, summing many independently-noised
    client updates (as FedAvg does) causes the noise to cancel out in
    expectation while each individual contribution stays private.
    """

    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    @property
    def scale(self) -> float:
        return self.sensitivity / self.epsilon

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        distribution = torch.distributions.Laplace(
            torch.zeros((), dtype=torch.float32), torch.full((), self.scale, dtype=torch.float32)
        )
        noise = distribution.sample(tensor.shape).to(tensor.dtype)
        return tensor + noise

    def privatize_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: self.add_noise(tensor) if torch.is_floating_point(tensor) else tensor.clone()
            for key, tensor in state_dict.items()
        }
