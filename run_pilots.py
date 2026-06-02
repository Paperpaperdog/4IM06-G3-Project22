#!/usr/bin/env python3
"""Run W3 pilot pipeline: subset prep -> Idea 1 -> Idea 2 -> comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_module(module: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, "-m", module, *(extra_args or [])]
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full W3 pilot pipeline.")
    parser.add_argument("--raise-dir", type=Path, default=None, help="RAISE folder (RAISE_1k.csv + tiff/).")
    parser.add_argument("--download", action="store_true", help="Download TIFFs from RAISE_1k.csv first.")
    parser.add_argument("--max-images", type=int, default=5, help="Limit images (pilots are slow).")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-idea1", action="store_true")
    parser.add_argument("--skip-idea2", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    prepare_args = ["--max-images", str(args.max_images)]
    if args.raise_dir is not None:
        prepare_args.extend(["--raise-dir", str(args.raise_dir)])
    if args.download:
        prepare_args.append("--download")

    if not args.skip_prepare:
        run_module("pilots.prepare_subset", prepare_args)
    if not args.skip_idea1:
        run_module("pilots.idea1_jpeg")
    if not args.skip_idea2:
        run_module("pilots.idea2_k_groups")
    if not args.skip_compare:
        run_module("pilots.compare")

    print("\nDone. See data/pilot_results/PILOT_SUMMARY.md")


if __name__ == "__main__":
    main()
