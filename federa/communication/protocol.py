"""msgpack codec for `Message` objects.

msgpack (rather than JSON) is used because `GlobalModel`/`GradientUpdate`
payloads embed raw serialized tensor bytes (see
`federa.models.serialization`); msgpack's native binary type carries those
without the ~33% size inflation of base64-in-JSON.
"""

from __future__ import annotations

import msgpack
from pydantic import TypeAdapter

from federa.communication.messages import Message

_adapter: TypeAdapter[Message] = TypeAdapter(Message)


def encode_message(message: Message) -> bytes:
    payload = message.model_dump(mode="python")
    packed: bytes = msgpack.packb(payload, use_bin_type=True)
    return packed


def decode_message(data: bytes) -> Message:
    raw = msgpack.unpackb(data, raw=False)
    return _adapter.validate_python(raw)
