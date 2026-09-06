from dataclasses import replace
from pathlib import Path

from microscan_ai.config import load_config
from microscan_ai.demo import make_demo_dataset
from microscan_ai.evaluation import evaluate
from microscan_ai.training import train


def test_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = make_demo_dataset(tmp_path / "demo", seed=11)
    config = load_config("configs/demo.yaml")
    config = replace(
        config,
        data=replace(config.data, root=data_root),
        training=replace(config.training, epochs=1, batch_size=18),
        output=replace(config.output, directory=tmp_path / "artifacts"),
    )
    train(config)
    checkpoint = config.output.directory / "best_model.pt"
    assert checkpoint.exists()
    results = evaluate(config, checkpoint)
    assert results["num_samples"] == 27
    assert 0.0 <= results["accuracy"] <= 1.0
