"""Unified compression API used by both `client.node` and `coordinator.server`.

`GradientUpdate.weights`/`GlobalModel.weights` (see
`federa.communication.messages`) always carry a `CompressedPayload` produced
here, so the wire format stays identical regardless of which scheme a
deployment picks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from federa.models.serialization import bytes_to_state_dict, state_dict_to_bytes
from federa.quantization.fp16 import state_dict_from_fp16, state_dict_to_fp16
from federa.quantization.int8 import Int8Tensor, dequantize_tensor_int8, quantize_tensor_int8

CompressionMethod = Literal["none", "int8", "fp16"]


@dataclass(slots=True)
class CompressedPayload:
    method: CompressionMethod
    data: bytes
    meta: dict[str, Any] = field(default_factory=dict)


def compress_state_dict(
    state_dict: dict[str, torch.Tensor], method: CompressionMethod = "none"
) -> CompressedPayload:
    if method == "none":
        return CompressedPayload(method="none", data=state_dict_to_bytes(state_dict))

    if method == "fp16":
        return CompressedPayload(
            method="fp16", data=state_dict_to_bytes(state_dict_to_fp16(state_dict))
        )

    if method == "int8":
        scales: dict[str, float] = {}
        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        packed: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            dtypes[key] = str(tensor.dtype)
            if torch.is_floating_point(tensor):
                quantized = quantize_tensor_int8(tensor)
                packed[key] = quantized.data
                scales[key] = quantized.scale
                shapes[key] = list(quantized.shape)
            else:
                packed[key] = tensor

        meta = {"scales": scales, "shapes": shapes, "dtypes": dtypes}
        return CompressedPayload(method="int8", data=state_dict_to_bytes(packed), meta=meta)

    raise ValueError(f"Unknown compression method: {method}")


def decompress_state_dict(payload: CompressedPayload) -> dict[str, torch.Tensor]:
    if payload.method == "none":
        return bytes_to_state_dict(payload.data)

    if payload.method == "fp16":
        return state_dict_from_fp16(bytes_to_state_dict(payload.data))

    if payload.method == "int8":
        packed = bytes_to_state_dict(payload.data)
        scales: dict[str, float] = payload.meta.get("scales", {})
        shapes: dict[str, list[int]] = payload.meta.get("shapes", {})
        dtypes: dict[str, str] = payload.meta.get("dtypes", {})

        result: dict[str, torch.Tensor] = {}
        for key, tensor in packed.items():
            if key in scales:
                quantized = Int8Tensor(data=tensor, scale=scales[key], shape=tuple(shapes[key]))
                value = dequantize_tensor_int8(quantized)
                dtype_name = dtypes.get(key, "torch.float32").rsplit(".", maxsplit=1)[-1]
                result[key] = value.to(getattr(torch, dtype_name))
            else:
                result[key] = tensor
        return result

    raise ValueError(f"Unknown compression method: {payload.method}")
