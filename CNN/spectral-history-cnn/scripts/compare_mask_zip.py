from __future__ import annotations

import argparse
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MaskStats:
    variant: str
    classes: int
    height: int
    width: int
    overlap_mean_offdiag: float
    overlap_min_offdiag: float
    overlap_max_offdiag: float
    mask_mean: float
    mask_std: float
    mask_range: float


def _extract_variant(path_in_zip: str) -> str:
    # .../outputs/<variant>/figures/masks/masks.npy
    m = re.search(r"/outputs/([^/]+)/figures/masks/masks\.npy$", path_in_zip)
    if not m:
        raise ValueError(f"Cannot parse variant from path: {path_in_zip}")
    return m.group(1)


def _load_npy_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=False)


def summarize_zip(zip_path: Path) -> list[MaskStats]:
    stats: list[MaskStats] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        mask_members = [m for m in members if m.endswith("/figures/masks/masks.npy")]

        for mask_member in sorted(mask_members):
            variant = _extract_variant(mask_member)
            overlap_member = mask_member.replace("/figures/masks/masks.npy", "/figures/mask_overlap.npy")
            if overlap_member not in members:
                continue

            masks = _load_npy_from_zip(zf, mask_member).astype(np.float64)
            overlap = _load_npy_from_zip(zf, overlap_member).astype(np.float64)

            if masks.ndim != 3:
                raise ValueError(f"Unexpected masks shape for {variant}: {masks.shape}")
            if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
                raise ValueError(f"Unexpected overlap shape for {variant}: {overlap.shape}")

            n = overlap.shape[0]
            offdiag = overlap[~np.eye(n, dtype=bool)]
            stats.append(
                MaskStats(
                    variant=variant,
                    classes=masks.shape[0],
                    height=masks.shape[1],
                    width=masks.shape[2],
                    overlap_mean_offdiag=float(offdiag.mean()),
                    overlap_min_offdiag=float(offdiag.min()),
                    overlap_max_offdiag=float(offdiag.max()),
                    mask_mean=float(masks.mean()),
                    mask_std=float(masks.std()),
                    mask_range=float(masks.max() - masks.min()),
                )
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mask quality across experiments packed in a zip.")
    parser.add_argument("--zip", required=True, help="Path to output_mask.zip")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    rows = summarize_zip(zip_path)
    if not rows:
        print("No masks.npy/mask_overlap.npy pairs found.")
        return

    rows = sorted(rows, key=lambda r: (r.overlap_mean_offdiag, -r.mask_range))

    print("Variant comparison (lower overlap_mean_offdiag is better):")
    print(
        "variant,classes,shape,overlap_mean_offdiag,overlap_min_offdiag,overlap_max_offdiag,"
        "mask_mean,mask_std,mask_range"
    )
    for r in rows:
        shape = f"{r.height}x{r.width}"
        print(
            f"{r.variant},{r.classes},{shape},{r.overlap_mean_offdiag:.6f},{r.overlap_min_offdiag:.6f},"
            f"{r.overlap_max_offdiag:.6f},{r.mask_mean:.6f},{r.mask_std:.6f},{r.mask_range:.6f}"
        )

    best = rows[0]
    print(
        f"\nRecommended by mask separability: {best.variant} "
        f"(mean off-diagonal overlap={best.overlap_mean_offdiag:.6f})"
    )


if __name__ == "__main__":
    main()

