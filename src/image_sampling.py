from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

VALID_EXTENSIONS = {".JPG", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(input_dir: str | Path) -> List[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    images = sorted(
        p for p in input_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
        and p.name.startswith("Faba-Seed-CC_Vf")
    )

    return images


def sample_images(images: Sequence[Path], sample_size: int = 30, seed: int = 42) -> List[Path]:
    if not images:
        raise ValueError("No matching images were found.")
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    rng = random.Random(seed)
    images = list(images)

    if len(images) <= sample_size:
        return sorted(images)

    return sorted(rng.sample(images, sample_size))
