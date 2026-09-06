from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config
from .model import build_model


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Config,
    classes: list[str],
    epoch: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "architecture": config.model.architecture,
            "width": config.model.width,
            "input_channels": config.data.channels,
            "image_size": config.data.image_size,
            "classes": classes,
            "seed": config.project.seed,
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    required = {
        "format_version",
        "model_state",
        "architecture",
        "width",
        "input_channels",
        "image_size",
        "classes",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    model = build_model(
        payload["architecture"],
        int(payload["input_channels"]),
        len(payload["classes"]),
        int(payload["width"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
