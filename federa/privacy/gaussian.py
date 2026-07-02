"""Gaussian-mechanism differential privacy.

Preferred over the Laplace mechanism when using an RDP/moments-accountant
(see `federa.privacy.accountant.rdp_epsilon`), since Gaussian noise composes
much more tightly across many training rounds.
"""

from __future__ import annotations

import math

import torch


class GaussianMechanism:
    """Adds N(0, sigma^2) noise calibrated for (epsilon, delta)-DP.

    Uses the classic analytic calibration from Dwork & Roth, Theorem A.1:
    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    @property
    def sigma(self) -> float:
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

    def privatize_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: self.add_noise(tensor) if torch.is_floating_point(tensor) else tensor.clone()
            for key, tensor in state_dict.items()
        }
