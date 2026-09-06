from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import __version__
from .checkpoints import save_checkpoint
from .config import Config
from .data import ImageFolderDataset, class_counts, validate_dataset
from .metrics import classification_metrics, collect_predictions
from .model import build_model
from .reproducibility import resolve_device, seed_everything


def _loader(dataset: ImageFolderDataset, config: Config, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.project.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        generator=generator,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment_metadata() -> dict[str, str]:
    return {
        "microscan_ai": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def train(config: Config) -> dict[str, object]:
    report = validate_dataset(config.data)
    if not report.valid:
        raise ValueError(f"Dataset validation failed: {report.to_dict()}")

    seed_everything(config.project.seed)
    device = resolve_device(config.training.device)
    train_data = ImageFolderDataset(config.data, "train", report.classes)
    val_data = ImageFolderDataset(config.data, "val", report.classes)
    train_loader = _loader(train_data, config, shuffle=True)
    val_loader = _loader(val_data, config, shuffle=False)

    model = build_model(
        config.model.architecture,
        config.data.channels,
        len(report.classes),
        config.model.width,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * inputs.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()

        val_targets, val_predictions, val_loss = collect_predictions(model, val_loader, device)
        val_metrics = classification_metrics(val_targets, val_predictions, report.classes)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / len(train_data),
            "train_accuracy": train_correct / len(train_data),
            "val_loss": val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                output_dir / "best_model.pt", model, config, report.classes, epoch, val_loss
            )

    run_manifest = {
        "project": config.project.name,
        "seed": config.project.seed,
        "device": str(device),
        "classes": report.classes,
        "class_counts": {
            "train": class_counts(train_data),
            "val": class_counts(val_data),
        },
        "dataset_validation": report.to_dict(),
        "config": asdict(config),
        "environment": environment_metadata(),
        "best_validation_loss": best_loss,
    }
    # Convert Paths from dataclass serialization before JSON output.
    run_manifest["config"]["data"]["root"] = str(config.data.root)  # type: ignore[index]
    run_manifest["config"]["output"]["directory"] = str(config.output.directory)  # type: ignore[index]
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest
