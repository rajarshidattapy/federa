"""QLoRA-ready hooks: 4-bit weight quantization + LoRA adapter injection.

This implements the QLoRA *pattern* -- a frozen 4-bit base model with small
trainable low-rank adapters, so federated rounds only ever need to transmit
tiny adapter deltas instead of the full model -- using a straightforward
symmetric block quantizer. It is not a drop-in replacement for bitsandbytes'
NF4 kernels; swap `quantize_tensor_int4`/`dequantize_tensor_int4` for
bitsandbytes if you need its exact numerics and fused CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class Int4Tensor:
    packed: torch.Tensor  # uint8, two 4-bit values packed per byte
    scale: float
    numel: int
    shape: tuple[int, ...]


def quantize_tensor_int4(tensor: torch.Tensor) -> Int4Tensor:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    max_abs = float(flat.abs().max()) if flat.numel() else 0.0
    scale = max_abs / 7.0 if max_abs > 0 else 1.0

    # Map signed range [-7, 7] to unsigned nibble range [1, 15].
    nibbles = (torch.clamp(torch.round(flat / scale), -7, 7) + 8).to(torch.uint8)
    if nibbles.numel() % 2 == 1:
        nibbles = torch.cat([nibbles, torch.zeros(1, dtype=torch.uint8)])

    packed = (nibbles[0::2] & 0x0F) | ((nibbles[1::2] & 0x0F) << 4)
    return Int4Tensor(packed=packed, scale=scale, numel=tensor.numel(), shape=tuple(tensor.shape))


def dequantize_tensor_int4(payload: Int4Tensor) -> torch.Tensor:
    low = payload.packed & 0x0F
    high = (payload.packed >> 4) & 0x0F

    interleaved = torch.empty(low.numel() * 2, dtype=torch.uint8)
    interleaved[0::2] = low
    interleaved[1::2] = high
    interleaved = interleaved[: payload.numel]

    values = interleaved.to(torch.int16) - 8
    return (values.to(torch.float32) * payload.scale).reshape(payload.shape)


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank adapter.

    Only `lora_a`/`lora_b` are trainable -- the base weight is frozen, which
    is what lets a QLoRA client ship a quantized frozen base model once and
    thereafter only train/transmit the tiny adapter deltas each round.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + self.scaling * lora_out


def inject_lora_adapters(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Recursively replaces every `nn.Linear` in-place with a `LoRALinear`."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora_adapters(child, rank=rank, alpha=alpha)
    return module


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Trainable adapter parameters only -- what a QLoRA client should train and send."""
    return [param for name, param in module.named_parameters() if "lora_" in name and param.requires_grad]
