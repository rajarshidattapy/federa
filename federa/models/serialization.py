"""Binary (de)serialization of model state dicts for network transmission.

State dicts are what actually crosses the wire between clients and the
coordinator (inside `GradientUpdate`/`GlobalModel` messages, see
`federa.communication.messages`), so this module is the single place that
knows how to turn `dict[str, torch.Tensor]` into `bytes` and back.
"""

from __future__ import annotations

import io

import torch


def state_dict_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(data: bytes, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    buffer = io.BytesIO(data)
    result: dict[str, torch.Tensor] = torch.load(
        buffer, map_location=map_location, weights_only=True
    )
    return result


def state_dict_num_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in state_dict.values())
