from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint
from .config import Config
from .data import ImageFolderDataset, validate_dataset
from .metrics import classification_metrics, collect_predictions
from .reproducibility import resolve_device, seed_everything
from .training import write_json


def evaluate(config: Config, checkpoint: str | Path) -> dict[str, object]:
    report = validate_dataset(config.data)
    if not report.valid:
        raise ValueError(f"Dataset validation failed: {report.to_dict()}")
    seed_everything(config.project.seed)
    device = resolve_device(config.training.device)
    model, payload = load_checkpoint(checkpoint, device)
    classes = list(payload["classes"])
    if classes != report.classes:
        raise ValueError(
            f"Checkpoint classes {classes} do not match dataset classes {report.classes}"
        )
    if int(payload["input_channels"]) != config.data.channels:
        raise ValueError("Checkpoint channels do not match configuration")
    if int(payload["image_size"]) != config.data.image_size:
        raise ValueError("Checkpoint image size does not match configuration")

    test_data = ImageFolderDataset(config.data, "test", classes)
    loader = DataLoader(
        test_data,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    targets, predictions, loss = collect_predictions(model, loader, device)
    results = classification_metrics(targets, predictions, classes)
    results.update(
        {
            "loss": loss,
            "num_samples": len(test_data),
            "checkpoint": str(checkpoint),
            "split": "test",
        }
    )
    write_json(config.output.directory / "test_metrics.json", results)
    return results
