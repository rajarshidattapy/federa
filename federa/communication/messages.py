"""Typed message schemas exchanged between clients and the coordinator.

Every message that crosses the wire is one of these six types. They form a
pydantic discriminated union on the `type` field, so `protocol.decode_message`
can deserialize a raw byte payload straight into the correct concrete class
with full validation.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    CLIENT_JOIN = "client_join"
    CLIENT_LEAVE = "client_leave"
    HEARTBEAT = "heartbeat"
    GLOBAL_MODEL = "global_model"
    GRADIENT_UPDATE = "gradient_update"
    TRAINING_METRICS = "training_metrics"


class ClientJoin(BaseModel):
    """Sent by a client immediately after the websocket handshake."""

    type: Literal[MessageType.CLIENT_JOIN] = MessageType.CLIENT_JOIN
    client_id: str
    num_samples: int = Field(ge=0)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ClientLeave(BaseModel):
    """Sent by a client (or synthesized by the coordinator) on disconnect."""

    type: Literal[MessageType.CLIENT_LEAVE] = MessageType.CLIENT_LEAVE
    client_id: str
    reason: str | None = None


class Heartbeat(BaseModel):
    """Periodic keep-alive in both directions, used to detect dead peers."""

    type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT
    client_id: str
    timestamp: float = Field(default_factory=time.time)


class GlobalModel(BaseModel):
    """Broadcast from the coordinator: the current global model weights."""

    type: Literal[MessageType.GLOBAL_MODEL] = MessageType.GLOBAL_MODEL
    round_number: int = Field(ge=0)
    weights: bytes
    quantization: str = "none"
    weights_meta: dict[str, Any] = Field(default_factory=dict)


class GradientUpdate(BaseModel):
    """A client's locally trained weights, sent back for aggregation."""

    type: Literal[MessageType.GRADIENT_UPDATE] = MessageType.GRADIENT_UPDATE
    client_id: str
    round_number: int = Field(ge=0)
    num_samples: int = Field(ge=0)
    weights: bytes
    quantization: str = "none"
    weights_meta: dict[str, Any] = Field(default_factory=dict)
    loss: float | None = None


class TrainingMetrics(BaseModel):
    """Reported alongside (or independently of) a GradientUpdate."""

    type: Literal[MessageType.TRAINING_METRICS] = MessageType.TRAINING_METRICS
    client_id: str
    round_number: int = Field(ge=0)
    loss: float
    accuracy: float | None = None
    duration_seconds: float | None = None


Message = Annotated[
    Union[
        ClientJoin,
        ClientLeave,
        Heartbeat,
        GlobalModel,
        GradientUpdate,
        TrainingMetrics,
    ],
    Field(discriminator="type"),
]
