from __future__ import annotations

import csv
import urllib.error
import urllib.request
from pathlib import Path


def default_csv_path(raise_dir: Path) -> Path | None:
    for name in ("RAISE_1k.csv", "RAISE_1K.csv", "raise_1k.csv"):
        path = raise_dir / name
        if path.is_file():
            return path
    return None


def load_raise_index(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_local_tiffs(raise_dir: Path) -> list[Path]:
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raise_dir.rglob(pattern))
    return sorted(set(files))


def download_tiffs_from_csv(
    csv_path: Path,
    out_dir: Path,
    max_images: int,
    skip_existing: bool = True,
) -> list[Path]:
    """Download TIFF files listed in RAISE_1k.csv (TIFF column URLs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_raise_index(csv_path)[:max_images]
    downloaded: list[Path] = []

    for index, entry in enumerate(entries, start=1):
        file_id = entry["File"].strip()
        url = entry["TIFF"].strip()
        dest = out_dir / f"{file_id}.TIF"

        if dest.is_file() and skip_existing:
            print(f"[{index}/{len(entries)}] skip existing {dest.name}")
            downloaded.append(dest)
            continue

        print(f"[{index}/{len(entries)}] download {file_id} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            print(f"  FAILED: {exc}")
            continue
        downloaded.append(dest)

    return downloaded
