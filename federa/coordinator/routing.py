"""Connected-client registry and broadcast fan-out.

A client that drops mid-broadcast (network blip, tab closed) shouldn't take
the whole round down -- `broadcast` swallows per-client send failures and
prunes the dead connection instead of propagating the exception.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from federa.communication.messages import Message
from federa.communication.websocket import MessageChannel
from federa.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ConnectedClient:
    client_id: str
    channel: MessageChannel
    num_samples: int = 0


class ClientRegistry:
    def __init__(self) -> None:
        self._clients: dict[str, ConnectedClient] = {}
        self._lock = asyncio.Lock()

    async def register(self, client: ConnectedClient) -> None:
        async with self._lock:
            self._clients[client.client_id] = client

    async def unregister(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)

    def get(self, client_id: str) -> ConnectedClient | None:
        return self._clients.get(client_id)

    def __len__(self) -> int:
        return len(self._clients)

    def client_ids(self) -> list[str]:
        return list(self._clients.keys())

    async def broadcast(self, message: Message) -> None:
        stale: list[str] = []
        for client_id, client in list(self._clients.items()):
            try:
                await client.channel.send(message)
            except Exception:
                logger.warning("broadcast_failed", extra={"client_id": client_id})
                stale.append(client_id)

        for client_id in stale:
            await self.unregister(client_id)

    async def close_all(self) -> None:
        """Cleanly closes every connection, e.g. once total_rounds is reached."""
        for client_id, client in list(self._clients.items()):
            try:
                await client.channel.close()
            except Exception:
                logger.warning("close_failed", extra={"client_id": client_id})
            await self.unregister(client_id)
