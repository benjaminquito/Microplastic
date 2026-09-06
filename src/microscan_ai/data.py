from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .config import DataConfig

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    target: int


@dataclass(frozen=True)
class ValidationReport:
    root: str
    classes: list[str]
    counts: dict[str, dict[str, int]]
    total_images: int
    duplicate_groups_across_splits: int
    invalid_files: list[str]

    @property
    def valid(self) -> bool:
        return not self.invalid_files and self.duplicate_groups_across_splits == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "classes": self.classes,
            "counts": self.counts,
            "total_images": self.total_images,
            "duplicate_groups_across_splits": self.duplicate_groups_across_splits,
            "invalid_files": self.invalid_files,
            "valid": self.valid,
        }


def discover_classes(root: Path) -> list[str]:
    split_classes: dict[str, list[str]] = {}
    for split in SPLITS:
        split_path = root / split
        if not split_path.is_dir():
            raise ValueError(f"Required split directory does not exist: {split_path}")
        split_classes[split] = sorted(path.name for path in split_path.iterdir() if path.is_dir())
        if len(split_classes[split]) < 2:
            raise ValueError(f"Split {split!r} must contain at least two class directories")
    expected = split_classes["train"]
    for split, classes in split_classes.items():
        if classes != expected:
            raise ValueError(f"Class directories differ: train={expected}, {split}={classes}")
    return expected


def image_paths(directory: Path, allowed_extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )


def validate_dataset(config: DataConfig) -> ValidationReport:
    classes = discover_classes(config.root)
    counts: dict[str, dict[str, int]] = {}
    invalid_files: list[str] = []
    hashes: dict[str, set[str]] = {}

    for split in SPLITS:
        counts[split] = {}
        for class_name in classes:
            paths = image_paths(config.root / split / class_name, config.allowed_extensions)
            counts[split][class_name] = len(paths)
            if not paths:
                invalid_files.append(f"{split}/{class_name}: no supported images")
            for path in paths:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError) as exc:
                    invalid_files.append(f"{path}: {exc}")
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, set()).add(split)

    duplicates = sum(1 for splits in hashes.values() if len(splits) > 1)
    total = sum(sum(per_class.values()) for per_class in counts.values())
    return ValidationReport(
        root=str(config.root),
        classes=classes,
        counts=counts,
        total_images=total,
        duplicate_groups_across_splits=duplicates,
        invalid_files=invalid_files,
    )


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, config: DataConfig, split: str, classes: list[str] | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.config = config
        self.split = split
        self.classes = classes or discover_classes(config.root)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[Sample] = []
        for name in self.classes:
            for path in image_paths(config.root / split / name, config.allowed_extensions):
                self.samples.append(Sample(path, self.class_to_index[name]))
        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        mode = "L" if self.config.channels == 1 else "RGB"
        with Image.open(sample.path) as image:
            image = image.convert(mode)
            image = image.resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        if self.config.channels == 1:
            array = array[np.newaxis, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array.copy()), sample.target


def class_counts(dataset: ImageFolderDataset) -> dict[str, int]:
    counts = Counter(sample.target for sample in dataset.samples)
    return {name: counts[index] for index, name in enumerate(dataset.classes)}
