from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpoints import load_checkpoint
from .reproducibility import resolve_device


def predict_image(
    checkpoint: str | Path, image_path: str | Path, device_name: str = "cpu", top_k: int = 3
) -> dict[str, object]:
    device = resolve_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    channels = int(payload["input_channels"])
    image_size = int(payload["image_size"])
    mode = "L" if channels == 1 else "RGB"
    with Image.open(image_path) as image:
        image = image.convert(mode).resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        array = array[np.newaxis, :, :]
    else:
        array = np.transpose(array, (2, 0, 1))
    inputs = torch.from_numpy(array.copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].cpu()
    classes = list(payload["classes"])
    k = min(max(1, top_k), len(classes))
    values, indices = torch.topk(probabilities, k)
    ranked = [
        {"class": classes[index], "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]
    return {
        "image": str(image_path),
        "predicted_class": ranked[0]["class"],
        "predictions": ranked,
        "warning": "Probabilities are model scores, not calibrated scientific confidence.",
    }
