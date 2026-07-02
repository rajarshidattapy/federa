"""FP16 (half precision) compression -- a simple 2x bandwidth reduction with
negligible accuracy loss, useful when Int8's extra quantization error isn't
worth the additional 2x saved.
"""

from __future__ import annotations

import torch


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16)


def from_fp16(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return tensor.to(dtype)


def state_dict_to_fp16(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: to_fp16(tensor) if torch.is_floating_point(tensor) else tensor.clone()
        for key, tensor in state_dict.items()
    }


def state_dict_from_fp16(
    state_dict: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    return {
        key: from_fp16(tensor, dtype) if tensor.dtype == torch.float16 else tensor.clone()
        for key, tensor in state_dict.items()
    }
