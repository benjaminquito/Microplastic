from pathlib import Path

from microscan_ai.config import DataConfig
from microscan_ai.data import ImageFolderDataset, validate_dataset
from microscan_ai.demo import make_demo_dataset


def _config(root: Path) -> DataConfig:
    return DataConfig(root=root, image_size=32, channels=1, allowed_extensions=(".png",))


def test_demo_dataset_is_valid_and_loadable(tmp_path: Path) -> None:
    root = make_demo_dataset(tmp_path / "demo", seed=7)
    report = validate_dataset(_config(root))
    assert report.valid
    assert report.total_images == 126
    assert report.classes == ["class_a", "class_b", "class_c"]
    image, target = ImageFolderDataset(_config(root), "train")[0]
    assert image.shape == (1, 32, 32)
    assert image.dtype.is_floating_point
    assert target in (0, 1, 2)


def test_duplicate_across_splits_is_detected(tmp_path: Path) -> None:
    root = make_demo_dataset(tmp_path / "demo", seed=7)
    source = root / "train" / "class_a" / "sample_000.png"
    destination = root / "test" / "class_a" / "copied.png"
    destination.write_bytes(source.read_bytes())
    report = validate_dataset(_config(root))
    assert not report.valid
    assert report.duplicate_groups_across_splits == 1
