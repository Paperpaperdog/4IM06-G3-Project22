import argparse
import csv
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(input_dir: Path) -> list[str]:
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return [str(p.relative_to(input_dir)) for p in sorted(paths)]


def list_csv_urls(input_csv: Path, url_column: str) -> list[str]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        urls = [row[url_column] for row in reader if row.get(url_column)]
    return urls


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir")
    source.add_argument("--input-csv")
    parser.add_argument("--url-column", default="TIFF")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        images = list_images(input_dir)
        source_info = {"type": "directory", "input_dir": str(input_dir)}
    else:
        input_csv = Path(args.input_csv)
        images = list_csv_urls(input_csv, args.url_column)
        source_info = {"type": "csv_urls", "input_csv": str(input_csv), "url_column": args.url_column}

    expected = args.train + args.val + args.test
    if len(images) < expected:
        raise ValueError(f"Need at least {expected} images, found {len(images)}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    split = {
        "seed": args.seed,
        "source": source_info,
        "train": images[: args.train],
        "val": images[args.train : args.train + args.val],
        "test": images[args.train + args.val : args.train + args.val + args.test],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


if __name__ == "__main__":
    main()
