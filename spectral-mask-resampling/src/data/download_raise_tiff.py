import argparse
import csv
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm


def list_urls(input_csv: Path, url_column: str) -> list[str]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row[url_column] for row in reader if row.get(url_column)]


def download_one(url: str, dst: Path, retries: int) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if dst.exists():
                dst.unlink()
    raise RuntimeError(f"Failed to download {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAISE TIFF files listed in a CSV.")
    parser.add_argument("--input-csv", default="../data/raise_raw/RAISE_1k.csv")
    parser.add_argument("--url-column", default="TIFF")
    parser.add_argument("--output-dir", default="data/raw/raise_tiff")
    parser.add_argument("--limit", type=int, help="Download only the first N URLs (for testing).")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = list_urls(input_csv, args.url_column)
    if args.limit is not None:
        urls = urls[: args.limit]

    skipped = 0
    downloaded = 0
    failed: list[str] = []

    for url in tqdm(urls, desc="download raise tiff"):
        filename = Path(urllib.parse.urlparse(url).path).name
        dst = output_dir / filename
        if dst.exists():
            skipped += 1
            continue
        try:
            download_one(url, dst, args.retries)
            downloaded += 1
        except RuntimeError:
            failed.append(url)

    print(f"done: downloaded={downloaded} skipped={skipped} failed={len(failed)} total={len(urls)}")
    if failed:
        print("failed urls:")
        for url in failed:
            print(url)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
