"""The public `Coordinator` API: a FastAPI/websocket FedAvg server.

Mirrors `Coordinator` from the TypeScript prototype's coordinator.ts -- the
central brain that never trains, only aggregates -- but drives a real
asyncio websocket server instead of a synchronous in-memory mock.
"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from federa.communication.messages import (
    ClientJoin,
    ClientLeave,
    GlobalModel,
    GradientUpdate,
    Heartbeat,
    TrainingMetrics,
)
from federa.communication.websocket import MessageChannel, StarletteWebSocketTransport
from federa.coordinator.aggregator import Aggregator
from federa.coordinator.routing import ClientRegistry, ConnectedClient
from federa.coordinator.state import GlobalState
from federa.models.base import FederatedModel
from federa.privacy.clipping import clip_state_dict_
from federa.quantization.compression import (
    CompressedPayload,
    compress_state_dict,
    decompress_state_dict,
)
from federa.training.checkpoint import save_checkpoint
from federa.training.fedavg import ClientUpdate
from federa.utils.config import CoordinatorSettings, load_coordinator_settings
from federa.utils.logging import configure_logging, get_logger
from federa.utils.metrics import RoundMetrics

logger = get_logger(__name__)


class Coordinator:
    """The central FedAvg server.

    ```python
    from federa import Coordinator
    from federa.models import wrap_model

    Coordinator(wrap_model(MyTorchModel())).run()
    ```
    """

    def __init__(
        self, model: FederatedModel, settings: CoordinatorSettings | None = None
    ) -> None:
        self.settings = settings or load_coordinator_settings()
        self.state = GlobalState(model=model, total_rounds=self.settings.rounds)
        self.registry = ClientRegistry()
        self.aggregator = Aggregator(self.settings)
        self.app = FastAPI(title="Federa Coordinator")
        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "ok",
                "round": self.state.round_number,
                "total_rounds": self.state.total_rounds,
                "connected_clients": len(self.registry),
            }

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            channel = MessageChannel(StarletteWebSocketTransport(websocket))
            client_id: str | None = None
            try:
                client_id = await self._handle_client(channel)
            except WebSocketDisconnect:
                pass
            finally:
                if client_id is not None:
                    await self.registry.unregister(client_id)
                    await self.registry.broadcast(ClientLeave(client_id=client_id))

    async def _handle_client(self, channel: MessageChannel) -> str:
        join = await channel.receive()
        if not isinstance(join, ClientJoin):
            raise ValueError(f"Expected ClientJoin as the first message, got {join.type!r}")

        client = ConnectedClient(
            client_id=join.client_id, channel=channel, num_samples=join.num_samples
        )
        await self.registry.register(client)
        logger.info(
            "client_joined", extra={"client_id": join.client_id, "num_samples": join.num_samples}
        )

        await channel.send(self._build_global_model_message())

        while True:
            try:
                message = await channel.receive()
            except (WebSocketDisconnect, RuntimeError):
                # RuntimeError: this connection (possibly this very one) was
                # just closed by `_run_round` -> `close_all()` because
                # training finished -- treat it the same as a clean
                # client-initiated disconnect.
                return join.client_id

            if isinstance(message, GradientUpdate):
                await self._handle_gradient_update(message)
            elif isinstance(message, Heartbeat):
                await channel.send(Heartbeat(client_id="coordinator"))
            elif isinstance(message, TrainingMetrics):
                self.state.metrics.record(
                    RoundMetrics(
                        round_number=message.round_number,
                        client_id=message.client_id,
                        num_samples=client.num_samples,
                        loss=message.loss,
                        accuracy=message.accuracy,
                        duration_seconds=message.duration_seconds,
                    )
                )
            elif isinstance(message, ClientLeave):
                return message.client_id

    def _build_global_model_message(self) -> GlobalModel:
        payload = compress_state_dict(
            self.state.model.get_weights(), self.settings.quantization.method
        )
        return GlobalModel(
            round_number=self.state.round_number,
            weights=payload.data,
            quantization=payload.method,
            weights_meta=payload.meta,
        )

    async def _handle_gradient_update(self, update_msg: GradientUpdate) -> None:
        payload = CompressedPayload(
            method=update_msg.quantization,  # type: ignore[arg-type]
            data=update_msg.weights,
            meta=update_msg.weights_meta,
        )
        weights = decompress_state_dict(payload)

        if self.settings.privacy.mechanism != "none":
            clip_state_dict_(weights, self.settings.privacy.max_grad_norm)

        async with self.state.lock:
            self.state.add_update(
                ClientUpdate(
                    client_id=update_msg.client_id,
                    num_samples=update_msg.num_samples,
                    weights=weights,
                )
            )
            if len(self.state.pending_updates) >= self.settings.min_clients_per_round:
                await self._run_round()

    async def _run_round(self) -> None:
        aggregated = self.aggregator.aggregate(self.state.pending_updates)
        self.state.model.set_weights(aggregated)
        completed_round = self.state.round_number
        self.state.clear_updates()
        self.state.round_number += 1

        summary = self.state.metrics.round_summary(completed_round)
        logger.info("round_complete", extra={"round": completed_round, **summary})

        save_checkpoint(self.settings.checkpoint_dir, self.state.round_number, aggregated, summary)

        if self.state.is_training_complete():
            logger.info("training_complete", extra={"total_rounds": self.state.total_rounds})
            await self.registry.close_all()
        else:
            await self.registry.broadcast(self._build_global_model_message())

    def run(self, *, host: str | None = None, port: int | None = None) -> None:
        """Blocking entrypoint that starts the uvicorn server."""
        configure_logging()
        uvicorn.run(self.app, host=host or self.settings.host, port=port or self.settings.port)


def main() -> None:
    """`federa-coordinator` console-script entrypoint.

    A pip-installed Coordinator needs a model to serve, so this dynamically
    imports one: `federa-coordinator myapp.models:build_model` where
    `build_model` is a zero-argument callable returning an `nn.Module`.
    """
    import argparse
    import importlib

    from federa.models.pytorch import wrap_model

    parser = argparse.ArgumentParser(prog="federa-coordinator")
    parser.add_argument(
        "model",
        help=(
            "Dotted path to a zero-arg callable returning an nn.Module, "
            "e.g. 'myapp.models:build_model'"
        ),
    )
    args = parser.parse_args()

    module_path, _, attr = args.model.partition(":")
    if not attr:
        raise SystemExit("Model must be specified as 'module.path:callable_name'")

    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    Coordinator(wrap_model(factory())).run()


if __name__ == "__main__":
    main()
