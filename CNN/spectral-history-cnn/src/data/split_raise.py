from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(input_dir: Path) -> list[Path]:
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split RAISE images by source file.")
    parser.add_argument("--input-dir", required=True, help="Directory containing RAISE-1k images.")
    parser.add_argument("--output-json", required=True, help="Path to write split JSON.")
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"RAISE directory does not exist: {input_dir}")

    images = list_images(input_dir)
    required = args.train + args.val + args.test
    if len(images) < required:
        raise ValueError(f"Need at least {required} images, found {len(images)} in {input_dir}.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    chosen = [images[i] for i in indices[:required]]

    train = chosen[: args.train]
    val = chosen[args.train : args.train + args.val]
    test = chosen[args.train + args.val :]

    def rel(paths: list[Path]) -> list[str]:
        return [str(p.relative_to(input_dir).as_posix()) for p in paths]

    save_json(
        {
            "seed": args.seed,
            "input_dir": str(input_dir),
            "num_images_found": len(images),
            "splits": {
                "train": rel(train),
                "val": rel(val),
                "test": rel(test),
            },
        },
        args.output_json,
    )
    print(f"Wrote split JSON to {args.output_json}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
