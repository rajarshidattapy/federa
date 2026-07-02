"""Round checkpointing for the global model."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    directory: str | Path,
    round_number: int,
    state_dict: dict[str, torch.Tensor],
    metrics: dict[str, float] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = directory / f"round_{round_number:05d}.pt"
    torch.save(state_dict, checkpoint_path)

    metadata_path = directory / f"round_{round_number:05d}.json"
    metadata_path.write_text(
        json.dumps({"round_number": round_number, "metrics": metrics or {}}, indent=2)
    )
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = torch.load(
        Path(path), map_location=map_location, weights_only=True
    )
    return result


def latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob("round_*.pt"))
    return checkpoints[-1] if checkpoints else None
