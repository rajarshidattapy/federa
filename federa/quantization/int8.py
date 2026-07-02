"""Symmetric 8-bit integer quantization.

Direct successor to `Quantizer.quantize`/`Quantizer.dequantize` in the
TypeScript prototype's quantization.ts: maps a float tensor's
[-max_abs, max_abs] range onto [-127, 127] for a 4x bandwidth reduction over
Float32, operating on `torch.Tensor` instead of the bespoke `Matrix` type.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Int8Tensor:
    data: torch.Tensor  # int8
    scale: float
    shape: tuple[int, ...]


def quantize_tensor_int8(tensor: torch.Tensor) -> Int8Tensor:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    max_abs = float(flat.abs().max()) if flat.numel() else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
    return Int8Tensor(data=quantized, scale=scale, shape=tuple(tensor.shape))


def dequantize_tensor_int8(payload: Int8Tensor) -> torch.Tensor:
    return (payload.data.to(torch.float32) * payload.scale).reshape(payload.shape)
