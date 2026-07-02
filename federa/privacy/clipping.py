"""Gradient/weight clipping.

Differential privacy noise is calibrated to a bounded *sensitivity* -- how
much a single client's update can change the aggregate. Clipping enforces
that bound before noise is added (see `federa.privacy.laplace`,
`federa.privacy.gaussian`) and before quantization.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def clip_state_dict_(state_dict: dict[str, torch.Tensor], max_norm: float) -> float:
    """In-place global L2-norm clipping across every floating-point tensor.

    Returns the pre-clip global norm.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    total_norm_sq = sum(
        float(torch.sum(tensor.detach().float() ** 2))
        for tensor in state_dict.values()
        if torch.is_floating_point(tensor)
    )
    total_norm = total_norm_sq**0.5
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for tensor in state_dict.values():
            if torch.is_floating_point(tensor):
                tensor.mul_(clip_coef)

    return total_norm


def clip_parameter_gradients_(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> float:
    """Clips `.grad` tensors in place before an optimizer step. Returns the pre-clip norm."""
    return float(torch.nn.utils.clip_grad_norm_(list(parameters), max_norm))
