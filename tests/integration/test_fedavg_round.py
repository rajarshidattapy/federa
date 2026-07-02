"""End-to-end test: two real SwarmNode clients, a real Coordinator, a real
websocket connection over a real TCP port -- no mocked transport.
"""

from __future__ import annotations

import asyncio
import socket

import torch
import torch.nn as nn
import uvicorn
from torch.utils.data import TensorDataset

from federa import Coordinator, SwarmNode
from federa.models.pytorch import wrap_model
from federa.utils.config import ClientSettings, CoordinatorSettings


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def test_full_fedavg_round_over_websocket(tmp_path):
    port = _free_port()
    settings = CoordinatorSettings(
        host="127.0.0.1",
        port=port,
        min_clients_per_round=2,
        rounds=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    coordinator = Coordinator(wrap_model(_TinyModel()), settings=settings)

    server = uvicorn.Server(
        uvicorn.Config(coordinator.app, host=settings.host, port=settings.port, log_level="error")
    )
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    try:
        dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 1))
        server_url = f"ws://127.0.0.1:{port}"

        node_a = SwarmNode(
            server_url,
            _TinyModel(),
            dataset,
            settings=ClientSettings(local_epochs=1, batch_size=4),
        )
        node_b = SwarmNode(
            server_url,
            _TinyModel(),
            dataset,
            settings=ClientSettings(local_epochs=1, batch_size=4),
        )

        await asyncio.wait_for(asyncio.gather(node_a._run(), node_b._run()), timeout=20)

        assert coordinator.state.round_number == 1
        assert (tmp_path / "checkpoints" / "round_00001.pt").exists()
    finally:
        server.should_exit = True
        await server_task
