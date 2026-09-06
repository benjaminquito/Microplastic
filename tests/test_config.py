from pathlib import Path

import pytest

from microscan_ai.config import load_config


def test_demo_config_loads() -> None:
    config = load_config(Path("configs/demo.yaml"))
    assert config.project.seed == 20260906
    assert config.data.channels == 1
    assert config.training.device == "cpu"


def test_invalid_channels_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
project: {name: test, seed: 1}
data: {root: data, image_size: 32, channels: 2, allowed_extensions: ['.png']}
model: {architecture: tiny_cnn, width: 8}
training:
  {epochs: 1, batch_size: 1, learning_rate: 0.1, weight_decay: 0,
   num_workers: 0, device: cpu}
output: {directory: artifacts/test}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="channels"):
        load_config(path)
