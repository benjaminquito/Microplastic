from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    image_size: int
    channels: int
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    width: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration value: {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")

    project = _required(raw, "project", "root")
    data = _required(raw, "data", "root")
    model = _required(raw, "model", "root")
    training = _required(raw, "training", "root")
    output = _required(raw, "output", "root")

    cfg = Config(
        project=ProjectConfig(
            name=str(_required(project, "name", "project")),
            seed=int(_required(project, "seed", "project")),
        ),
        data=DataConfig(
            root=Path(_required(data, "root", "data")),
            image_size=int(_required(data, "image_size", "data")),
            channels=int(_required(data, "channels", "data")),
            allowed_extensions=tuple(
                str(item).lower() for item in _required(data, "allowed_extensions", "data")
            ),
        ),
        model=ModelConfig(
            architecture=str(_required(model, "architecture", "model")),
            width=int(_required(model, "width", "model")),
        ),
        training=TrainingConfig(
            epochs=int(_required(training, "epochs", "training")),
            batch_size=int(_required(training, "batch_size", "training")),
            learning_rate=float(_required(training, "learning_rate", "training")),
            weight_decay=float(_required(training, "weight_decay", "training")),
            num_workers=int(_required(training, "num_workers", "training")),
            device=str(_required(training, "device", "training")),
        ),
        output=OutputConfig(directory=Path(_required(output, "directory", "output"))),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.data.image_size < 16:
        raise ValueError("data.image_size must be at least 16")
    if cfg.data.channels not in (1, 3):
        raise ValueError("data.channels must be 1 or 3")
    if cfg.model.architecture != "tiny_cnn":
        raise ValueError("Only model.architecture=tiny_cnn is currently supported")
    if cfg.model.width < 1:
        raise ValueError("model.width must be positive")
    if cfg.training.epochs < 1 or cfg.training.batch_size < 1:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if cfg.training.learning_rate <= 0 or cfg.training.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters")
    if cfg.training.num_workers < 0:
        raise ValueError("training.num_workers cannot be negative")
    if cfg.training.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("training.device must be auto, cpu, cuda, or mps")
