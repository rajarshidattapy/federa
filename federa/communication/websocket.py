"""Transport-agnostic message channel over a websocket.

The coordinator speaks to Starlette's `WebSocket` server-side object while
clients speak to a `websockets` client connection -- two unrelated APIs.
`MessageChannel` hides that behind one `RawTransport` protocol so
`federa.communication.protocol` (and everything built on it) only has to
know about `Message` objects, not which library produced the socket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from federa.communication.messages import Message
from federa.communication.protocol import decode_message, encode_message

if TYPE_CHECKING:
    from starlette.websockets import WebSocket
    from websockets.asyncio.client import ClientConnection


@runtime_checkable
class RawTransport(Protocol):
    async def send_raw(self, data: bytes) -> None: ...
    async def receive_raw(self) -> bytes: ...
    async def close_raw(self) -> None: ...


class StarletteWebSocketTransport:
    """Adapts a FastAPI/Starlette server-side `WebSocket` to `RawTransport`."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send_raw(self, data: bytes) -> None:
        await self._websocket.send_bytes(data)

    async def receive_raw(self) -> bytes:
        data: bytes = await self._websocket.receive_bytes()
        return data

    async def close_raw(self) -> None:
        await self._websocket.close()


class ClientWebSocketTransport:
    """Adapts a `websockets` client connection to `RawTransport`."""

    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection

    async def send_raw(self, data: bytes) -> None:
        await self._connection.send(data)

    async def receive_raw(self) -> bytes:
        data = await self._connection.recv()
        if isinstance(data, str):
            return data.encode()
        return bytes(data)

    async def close_raw(self) -> None:
        await self._connection.close()


class MessageChannel:
    """Send/receive typed `Message` objects over any `RawTransport`."""

    def __init__(self, transport: RawTransport) -> None:
        self._transport = transport

    async def send(self, message: Message) -> None:
        await self._transport.send_raw(encode_message(message))

    async def receive(self) -> Message:
        raw = await self._transport.receive_raw()
        return decode_message(raw)

    async def close(self) -> None:
        await self._transport.close_raw()
