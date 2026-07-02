"""Manages the client's websocket connection lifecycle, including reconnect/backoff."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import websockets

from federa.communication.websocket import ClientWebSocketTransport, MessageChannel
from federa.utils.logging import get_logger

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection as WSClientConnection

logger = get_logger(__name__)


class ClientConnection:
    def __init__(
        self,
        server_url: str,
        *,
        reconnect_backoff_seconds: float = 2.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self.server_url = server_url
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.max_reconnect_attempts = max_reconnect_attempts
        self._channel: MessageChannel | None = None
        self._raw_connection: WSClientConnection | None = None

    async def connect(self) -> MessageChannel:
        attempt = 0
        while True:
            try:
                self._raw_connection = await websockets.connect(self.server_url, max_size=None)
                self._channel = MessageChannel(ClientWebSocketTransport(self._raw_connection))
                logger.info("connected", extra={"server_url": self.server_url})
                return self._channel
            except OSError as exc:
                attempt += 1
                if attempt > self.max_reconnect_attempts:
                    raise ConnectionError(
                        f"Failed to connect to {self.server_url} after {attempt} attempts"
                    ) from exc
                delay = self.reconnect_backoff_seconds * attempt
                logger.warning(
                    "connect_failed_retrying",
                    extra={"attempt": attempt, "delay_seconds": delay, "error": str(exc)},
                )
                await asyncio.sleep(delay)

    @property
    def channel(self) -> MessageChannel:
        if self._channel is None:
            raise RuntimeError("Not connected; call connect() first")
        return self._channel

    async def close(self) -> None:
        if self._raw_connection is not None:
            await self._raw_connection.close()
        self._raw_connection = None
        self._channel = None
