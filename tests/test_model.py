import torch

from microscan_ai.model import build_model


def test_model_output_shape_and_backward_pass() -> None:
    model = build_model("tiny_cnn", input_channels=1, num_classes=3, width=4)
    inputs = torch.rand(2, 1, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (2, 3)
    outputs.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
