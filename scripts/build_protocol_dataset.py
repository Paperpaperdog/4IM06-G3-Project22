#!/usr/bin/env python3
"""
Build protocol dataset for the a->c vs b->c study.

Pipeline:
1) Read square PNG images (output of preprocess_raise.py)
2) Auto-generate reference sizes from target size + peak sizes
3) For each image and reference size:
   a) Resize to reference size
   b) Resize again to target size
4) Save generated images and metadata CSV
5) Optionally call an external processing script (e.g., demo/ird) per sample
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image


INTERP_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def list_pngs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.png") if p.is_file())


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def generate_reference_sizes(
    target: int,
    peak_sizes: Iterable[int],
    k_offsets: Iterable[int],
    num_refs_per_peak: int,
    min_ref: int,
    max_ref: int,
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for peak in peak_sizes:
        candidates = set()
        for k in k_offsets:
            candidates.add(peak + k)
            candidates.add(target - peak + k)

        valid = sorted(x for x in candidates if min_ref <= x <= max_ref)
        if not valid:
            result[peak] = []
            continue

        if len(valid) <= num_refs_per_peak:
            result[peak] = valid
        else:
            if num_refs_per_peak <= 1:
                # Single reference requested: pick middle candidate for stability.
                result[peak] = [valid[len(valid) // 2]]
                continue
            # Pick evenly across sorted candidates to keep variety.
            idxs = []
            for i in range(num_refs_per_peak):
                idx = round(i * (len(valid) - 1) / (num_refs_per_peak - 1))
                idxs.append(idx)
            picked = sorted({valid[i] for i in idxs})
            # Ensure exactly num_refs_per_peak where possible.
            if len(picked) < num_refs_per_peak:
                for v in valid:
                    if v not in picked:
                        picked.append(v)
                    if len(picked) == num_refs_per_peak:
                        break
                picked.sort()
            result[peak] = picked[:num_refs_per_peak]
    return result


def ensure_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def run_external_processor(command_template: str, image_path: Path, output_dir: Path) -> None:
    cmd = command_template.format(image=str(image_path), outdir=str(output_dir))
    subprocess.run(cmd, shell=True, check=True)

def run_ird_processor(
    ird_root: Path,
    image_path: Path,
    output_dir: Path,
    direction: str,
    preproc: str,
    is_jpeg: bool,
) -> None:
    detect_script = ird_root / "detect_one_image.py"
    cmd = [
        sys.executable,
        str(detect_script),
        str(image_path.resolve()),
        "--direction",
        direction,
        "--preproc",
        preproc,
        "--out_folder",
        str(output_dir.resolve()),
    ]
    if is_jpeg:
        cmd.append("--is_jpeg")
    subprocess.run(cmd, check=True, cwd=str(ird_root))


def process_dataset(args: argparse.Namespace) -> None:
    in_root: Path = args.input_dir
    out_root: Path = args.output_dir
    target = args.target_size
    peaks = parse_int_list(args.peak_sizes)
    k_offsets = parse_int_list(args.k_offsets)
    interp = INTERP_MAP[args.interpolation]

    refs_by_peak = generate_reference_sizes(
        target=target,
        peak_sizes=peaks,
        k_offsets=k_offsets,
        num_refs_per_peak=args.num_refs_per_peak,
        min_ref=args.min_ref_size,
        max_ref=args.max_ref_size,
    )

    images = list_pngs(in_root)
    if not images:
        print("No PNG images found. Run preprocess_raise.py first.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    sample_dir = out_root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = out_root / "external_outputs"
    ext_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_root / "metadata.csv"
    plan_path = out_root / "size_plan.csv"

    with plan_path.open("w", newline="", encoding="utf-8") as f_plan:
        w = csv.writer(f_plan)
        w.writerow(["peak_size", "reference_sizes"])
        for peak, refs in refs_by_peak.items():
            w.writerow([peak, ";".join(str(r) for r in refs)])

    rows = []
    ok = 0
    failed = 0
    sample_id = 0
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
                img = ensure_square(img)
                src_side = img.size[0]

                for peak in peaks:
                    refs = refs_by_peak.get(peak, [])
                    for ref in refs:
                        step1 = img.resize((ref, ref), resample=interp)
                        step2 = step1.resize((target, target), resample=interp)

                        rel_name = img_path.stem
                        out_name = f"{sample_id:08d}_{rel_name}_p{peak}_r{ref}_t{target}.png"
                        out_img = sample_dir / out_name
                        step2.save(out_img, format="PNG")

                        sample_result_dir = ext_dir / f"{sample_id:08d}"
                        sample_result_dir.mkdir(parents=True, exist_ok=True)

                        if args.ird_root:
                            try:
                                run_ird_processor(
                                    ird_root=args.ird_root,
                                    image_path=out_img,
                                    output_dir=sample_result_dir,
                                    direction=args.ird_direction,
                                    preproc=args.ird_preproc,
                                    is_jpeg=args.ird_is_jpeg,
                                )
                            except Exception as exc:  # noqa: BLE001
                                print(f"[WARN] IRD processor failed for {out_img.name}: {exc}")

                        if args.external_command:
                            try:
                                run_external_processor(args.external_command, out_img, sample_result_dir)
                            except Exception as exc:  # noqa: BLE001
                                print(f"[WARN] external processor failed for {out_img.name}: {exc}")

                        rows.append(
                            {
                                "sample_id": sample_id,
                                "source_image": str(img_path),
                                "source_square_size": src_side,
                                "peak_size": peak,
                                "reference_size": ref,
                                "target_size": target,
                                "interpolation": args.interpolation,
                                "output_image": str(out_img),
                            }
                        )
                        sample_id += 1
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {img_path}: {exc}")

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "source_image",
            "source_square_size",
            "peak_size",
            "reference_size",
            "target_size",
            "interpolation",
            "output_image",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Processed source images: {ok}, Failed source images: {failed}")
    print(f"Generated samples: {len(rows)}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved size plan: {plan_path}")
    if args.ird_root:
        print(f"IRD outputs root: {ext_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build resize protocol dataset automatically.")
    p.add_argument("--input-dir", type=Path, required=True, help="Square PNG input directory")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--target-size", type=int, default=384, help="Final target square size")
    p.add_argument(
        "--peak-sizes",
        type=str,
        default="64,85,96",
        help="Comma-separated peak/reference seed sizes",
    )
    p.add_argument(
        "--k-offsets",
        type=str,
        default="0,1,2",
        help="Comma-separated k offsets",
    )
    p.add_argument("--num-refs-per-peak", type=int, default=5, help="Reference sizes per peak")
    p.add_argument("--min-ref-size", type=int, default=32, help="Minimum valid reference size")
    p.add_argument("--max-ref-size", type=int, default=4096, help="Maximum valid reference size")
    p.add_argument(
        "--interpolation",
        choices=sorted(INTERP_MAP.keys()),
        default="bicubic",
        help="Resize interpolation method",
    )
    p.add_argument(
        "--external-command",
        type=str,
        default="",
        help=(
            "Optional command template to run per generated sample. "
            "Use {image} and {outdir} placeholders."
        ),
    )
    p.add_argument(
        "--ird-root",
        type=Path,
        default=None,
        help=(
            "Optional path to demo/ird folder. "
            "If set, detect_one_image.py will run automatically per generated sample."
        ),
    )
    p.add_argument(
        "--ird-direction",
        choices=["h", "v", "both"],
        default="both",
        help="Direction passed to detect_one_image.py",
    )
    p.add_argument(
        "--ird-preproc",
        choices=["rt", "tv", "dct", "phot", "none"],
        default="rt",
        help="Preprocessing passed to detect_one_image.py",
    )
    p.add_argument(
        "--ird-is-jpeg",
        action="store_true",
        help="Pass --is_jpeg to detect_one_image.py",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
