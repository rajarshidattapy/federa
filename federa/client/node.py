"""The public `SwarmNode` client API.

Successor to `SwarmNode` in the TypeScript prototype's client-node.ts.
Where the original wrapped a fixed feed-forward `NeuralNetwork` and mocked
the websocket connection, this wraps any `torch.nn.Module`/`Dataset` pair
and drives a real asyncio connection to a `federa.Coordinator`.
"""

from __future__ import annotations

import asyncio
import uuid
from urllib.parse import urlsplit, urlunsplit

import torch.nn as nn
import websockets.exceptions
from torch.utils.data import DataLoader, Dataset

from federa.client.scheduler import PrivacyHook, TrainingScheduler
from federa.client.trainer import LocalTrainer
from federa.client.websocket import ClientConnection
from federa.communication.messages import ClientJoin, ClientLeave
from federa.models.base import FederatedModel
from federa.models.pytorch import TorchModelAdapter
from federa.privacy.gaussian import GaussianMechanism
from federa.privacy.laplace import LaplaceMechanism
from federa.training.optimizer import OptimizerName
from federa.utils.config import ClientSettings, load_client_settings
from federa.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _normalize_server_url(url: str) -> str:
    """`ws://host:port` -> `ws://host:port/ws`; explicit paths are left alone."""
    parts = urlsplit(url)
    path = parts.path if parts.path not in ("", "/") else "/ws"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


class SwarmNode:
    """The client-side node.

    ```python
    from federa import SwarmNode

    node = SwarmNode(server="ws://localhost:8000", model=model, dataset=dataset)
    node.start_training()
    ```
    """

    def __init__(
        self,
        server: str,
        model: nn.Module | FederatedModel,
        dataset: Dataset,
        *,
        client_id: str | None = None,
        settings: ClientSettings | None = None,
        optimizer_name: OptimizerName = "sgd",
        fedprox_mu: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id or str(uuid.uuid4())
        self.settings = settings or load_client_settings()
        self.settings.server_url = _normalize_server_url(server)

        self.model: FederatedModel = (
            model if isinstance(model, FederatedModel) else TorchModelAdapter(model)
        )
        self.dataset = dataset
        self.optimizer_name = optimizer_name
        self.fedprox_mu = fedprox_mu
        self.device = device

        self._connection = ClientConnection(
            self.settings.server_url,
            reconnect_backoff_seconds=self.settings.reconnect_backoff_seconds,
            max_reconnect_attempts=self.settings.max_reconnect_attempts,
        )
        self._privacy_hook = self._build_privacy_hook()
        self.is_connected = False
        self.is_training = False

    def _build_privacy_hook(self) -> PrivacyHook | None:
        privacy = self.settings.privacy
        if privacy.mechanism == "laplace":
            return LaplaceMechanism(privacy.epsilon, privacy.sensitivity).privatize_state_dict
        if privacy.mechanism == "gaussian":
            return GaussianMechanism(
                privacy.epsilon, privacy.delta, privacy.sensitivity
            ).privatize_state_dict
        return None

    def start_training(self) -> None:
        """Blocking entrypoint: connects, joins, and trains until the coordinator
        signals training is complete (by closing the connection)."""
        configure_logging()
        asyncio.run(self._run())

    async def _run(self) -> None:
        dataloader = DataLoader(self.dataset, batch_size=self.settings.batch_size, shuffle=True)
        trainer = LocalTrainer(
            self.model,
            dataloader,
            settings=self.settings,
            optimizer_name=self.optimizer_name,
            fedprox_mu=self.fedprox_mu,
            device=self.device,
        )

        channel = await self._connection.connect()
        self.is_connected = True
        num_samples = len(self.dataset)  # type: ignore[arg-type]
        await channel.send(ClientJoin(client_id=self.client_id, num_samples=num_samples))
        logger.info("joined", extra={"client_id": self.client_id, "num_samples": num_samples})

        scheduler = TrainingScheduler(self.client_id, channel, trainer, self.settings, self._privacy_hook)
        self.is_training = True
        try:
            await scheduler.run()
        except websockets.exceptions.ConnectionClosed:
            logger.info("training_complete", extra={"client_id": self.client_id})
        finally:
            self.is_training = False
            self.is_connected = False
            try:
                await channel.send(ClientLeave(client_id=self.client_id))
            except Exception:
                pass
            await self._connection.close()
