from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(input_dir: Path) -> list[Path]:
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def split_sequence(items: list, train: int, val: int, test: int, seed: int) -> tuple[list, list, list]:
    required = train + val + test
    if len(items) < required:
        raise ValueError(f"Need at least {required} sources, found {len(items)}.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    chosen = [items[i] for i in indices[:required]]
    train_items = chosen[:train]
    val_items = chosen[train : train + val]
    test_items = chosen[train + val :]
    return train_items, val_items, test_items


def read_csv_sources(csv_path: Path, id_column: str, url_column: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    missing = [col for col in (id_column, url_column) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in {csv_path}: {missing}. Available columns: {list(df.columns)}")

    sources = []
    for csv_index, row in df.iterrows():
        file_id = str(row[id_column]).strip()
        url = str(row[url_column]).strip()
        if not file_id or not url or url.lower() == "nan":
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        sources.append(
            {
                "file_id": file_id,
                "url": url,
                "url_column": url_column,
                "csv_index": int(csv_index),
            }
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Split RAISE images by source file.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-dir", help="Directory containing already downloaded RAISE images.")
    source_group.add_argument("--csv", help="RAISE CSV containing image URLs.")
    parser.add_argument("--output-json", required=True, help="Path to write split JSON.")
    parser.add_argument("--id-column", default="File", help="CSV column containing the source image id.")
    parser.add_argument("--url-column", default="TIFF", help="CSV URL column to use, usually TIFF.")
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.csv:
        csv_path_for_reading = Path(args.csv).expanduser().resolve()
        if not csv_path_for_reading.exists():
            raise FileNotFoundError(f"RAISE CSV does not exist: {csv_path_for_reading}")
        sources = read_csv_sources(csv_path_for_reading, args.id_column, args.url_column)
        train, val, test = split_sequence(sources, args.train, args.val, args.test, args.seed)
        save_json(
            {
                "seed": args.seed,
                "source_type": "csv_url",
                "csv_path": Path(args.csv).as_posix(),
                "id_column": args.id_column,
                "url_column": args.url_column,
                "num_sources_found": len(sources),
                "splits": {
                    "train": train,
                    "val": val,
                    "test": test,
                },
            },
            args.output_json,
        )
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"RAISE directory does not exist: {input_dir}")

        images = list_images(input_dir)
        train, val, test = split_sequence(images, args.train, args.val, args.test, args.seed)

        def rel(paths: list[Path]) -> list[str]:
            return [str(p.relative_to(input_dir).as_posix()) for p in paths]

        save_json(
            {
                "seed": args.seed,
                "source_type": "local_dir",
                "input_dir": str(input_dir),
                "num_sources_found": len(images),
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
