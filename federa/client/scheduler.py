"""Drives the client-side federated loop: receive global model -> train -> send update -> repeat."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import torch

from federa.client.trainer import LocalTrainer
from federa.communication.messages import GlobalModel, GradientUpdate, Heartbeat, TrainingMetrics
from federa.communication.websocket import MessageChannel
from federa.quantization.compression import (
    CompressedPayload,
    compress_state_dict,
    decompress_state_dict,
)
from federa.utils.config import ClientSettings
from federa.utils.logging import get_logger

logger = get_logger(__name__)

PrivacyHook = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


class TrainingScheduler:
    def __init__(
        self,
        client_id: str,
        channel: MessageChannel,
        trainer: LocalTrainer,
        settings: ClientSettings,
        privacy_hook: PrivacyHook | None = None,
    ) -> None:
        self.client_id = client_id
        self.channel = channel
        self.trainer = trainer
        self.settings = settings
        self._privacy_hook = privacy_hook
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            while True:
                message = await self.channel.receive()
                if isinstance(message, GlobalModel):
                    await self._handle_global_model(message)
        finally:
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()

    async def _handle_global_model(self, message: GlobalModel) -> None:
        payload = CompressedPayload(
            method=message.quantization,  # type: ignore[arg-type]
            data=message.weights,
            meta=message.weights_meta,
        )
        self.trainer.model.set_weights(decompress_state_dict(payload))

        result = self.trainer.train_round(epochs=self.settings.local_epochs)

        weights_to_send = result.weights
        if self._privacy_hook is not None:
            weights_to_send = self._privacy_hook(weights_to_send)

        compressed = compress_state_dict(weights_to_send, self.settings.quantization.method)
        await self.channel.send(
            GradientUpdate(
                client_id=self.client_id,
                round_number=message.round_number,
                num_samples=result.num_samples,
                weights=compressed.data,
                quantization=compressed.method,
                weights_meta=compressed.meta,
                loss=result.loss,
            )
        )
        await self.channel.send(
            TrainingMetrics(
                client_id=self.client_id,
                round_number=message.round_number,
                loss=result.loss,
                duration_seconds=result.duration_seconds,
            )
        )
        logger.info(
            "round_trained",
            extra={"round": message.round_number, "loss": result.loss, "client_id": self.client_id},
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_seconds)
            try:
                await self.channel.send(Heartbeat(client_id=self.client_id))
            except Exception:
                return
