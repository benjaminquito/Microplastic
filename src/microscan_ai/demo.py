from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def _draw_sample(class_index: int, sample_index: int, seed: int, size: int) -> Image.Image:
    rng = random.Random(seed + class_index * 100_000 + sample_index)
    image = Image.new("L", (size, size), color=rng.randint(0, 20))
    draw = ImageDraw.Draw(image)
    foreground = rng.randint(205, 255)
    jitter_x = rng.randint(-3, 3)
    jitter_y = rng.randint(-3, 3)
    center = size // 2
    if class_index == 0:
        x = center + jitter_x
        draw.line((x, 6, x + rng.randint(-2, 2), size - 7), fill=foreground, width=3)
    elif class_index == 1:
        y = center + jitter_y
        draw.line((6, y, size - 7, y + rng.randint(-2, 2)), fill=foreground, width=3)
    else:
        radius = rng.randint(6, 9)
        x = center + jitter_x
        y = center + jitter_y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=foreground, width=3)

    pixels = image.load()
    for _ in range(size * size // 16):
        x = rng.randrange(size)
        y = rng.randrange(size)
        pixels[x, y] = min(255, max(0, pixels[x, y] + rng.randint(-15, 25)))
    return image


def make_demo_dataset(output: str | Path, seed: int = 20260906, force: bool = False) -> Path:
    output_path = Path(output)
    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"Output already exists: {output_path}; pass --force to replace it"
            )
        shutil.rmtree(output_path)
    counts = {"train": 24, "val": 9, "test": 9}
    classes = ["class_a", "class_b", "class_c"]
    for split_index, (split, count) in enumerate(counts.items()):
        for class_index, class_name in enumerate(classes):
            directory = output_path / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for sample_index in range(count):
                image = _draw_sample(
                    class_index,
                    sample_index + split_index * 10_000,
                    seed,
                    32,
                )
                image.save(directory / f"sample_{sample_index:03d}.png")
    return output_path
