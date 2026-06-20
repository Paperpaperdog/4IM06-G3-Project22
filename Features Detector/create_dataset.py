import argparse
import io
from pathlib import Path
from typing import List, Optional

from PIL import Image


# ============================================================
# Dataset post-processing script
# ============================================================
# This script creates forensic post-processing versions:
#
# output_dir/
#     original/          optional
#     jpeg/
#     upsample_x8/
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESAMPLE_PERIOD = 8


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def list_images(input_dir: str) -> List[Path]:
    input_dir = Path(input_dir)
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w8 = w - w % RESAMPLE_PERIOD
    h8 = h - h % RESAMPLE_PERIOD

    if w8 < RESAMPLE_PERIOD or h8 < RESAMPLE_PERIOD:
        raise ValueError(f"Image too small after crop: {w}x{h}")

    return img.crop((0, 0, w8, h8))


def top_left_crop_base(img: Image.Image, base_crop_size: Optional[int]) -> Image.Image:
    """
    Top-left-crop the input before generating all versions.

    This prevents upsample_x8 from becoming too large.
    The crop size is forced to a multiple of 8.
    """
    img = img.convert("RGB")

    if base_crop_size is None or base_crop_size <= 0:
        return crop_to_multiple_of_8(img)

    w, h = img.size
    crop = min(w, h, base_crop_size)
    crop = crop - crop % RESAMPLE_PERIOD

    if crop < RESAMPLE_PERIOD:
        raise ValueError(f"Image too small for base crop: {w}x{h}")

    left = 0
    top = 0
    return img.crop((left, top, left + crop, top + crop))


def save_png(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def jpeg_compress_decode(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def factor8_upsample(img: Image.Image) -> Image.Image:
    img = crop_to_multiple_of_8(img.convert("RGB"))
    w, h = img.size

    up = img.resize(
        (w * RESAMPLE_PERIOD, h * RESAMPLE_PERIOD),
        RESAMPLE_BICUBIC
    )

    return up.convert("RGB")


def make_versions(img: Image.Image, quality: int, base_crop_size: Optional[int]) -> dict:
    """
    Generate requested post-processing versions from the same cropped base image.

    original:
        top-left-cropped original image, optional output
    jpeg:
        I -> JPEG(I)
    downsample_x8:
        I -> bicubic downsample by factor 8
    upsample_x8:
        I -> bicubic upsample by factor 8
    """
    img = top_left_crop_base(img, base_crop_size)

    versions = {
        "original": img,
        "jpeg": jpeg_compress_decode(img, quality=quality),
        "upsample_x8": factor8_upsample(img),
    }
    return versions


def main():
    parser = argparse.ArgumentParser(
        description="Create JPEG and bicubic upsample x8 versions of an image dataset."
    )

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--include_original", action="store_true")
    parser.add_argument(
        "--base_crop_size",
        type=int,
        default=256,
        help="Top-left-crop input images to this size before processing. Use 0 to disable. Recommended: 256 or 512."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    paths = list_images(input_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No valid images found in {input_dir}")

    base_crop_size = None if args.base_crop_size <= 0 else args.base_crop_size

    print(f"[Info] Found {len(paths)} images.")
    print(f"[Info] JPEG quality: {args.quality}")
    print(f"[Info] Base top-left crop size: {base_crop_size if base_crop_size is not None else 'disabled'}")
    print("[Info] Upsample class: x8 upsampling with bicubic interpolation")

    if args.include_original:
        (output_dir / "original").mkdir(parents=True, exist_ok=True)
    (output_dir / "jpeg").mkdir(parents=True, exist_ok=True)
    (output_dir / "upsample_x8").mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for idx, path in enumerate(paths):
        try:
            img = open_rgb(path)
            versions = make_versions(
                img=img,
                quality=args.quality,
                base_crop_size=base_crop_size,
            )

            stem = path.stem

            if args.include_original:
                save_png(versions["original"], output_dir / "original" / f"{stem}_original_crop{args.base_crop_size}_p8.png")

            save_png(versions["jpeg"], output_dir / "jpeg" / f"{stem}_jpeg_q{args.quality}.png")
            save_png(versions["upsample_x8"], output_dir / "upsample_x8" / f"{stem}_upsample_x8_bicubic.png")

            processed += 1

        except Exception as e:
            skipped += 1
            print(f"[Warning] Skipped {path.name}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"[Info] Processed {idx + 1}/{len(paths)} input images.")

    print("\n[Done] Dataset post-processing finished.")
    print(f"[Done] Processed images: {processed}")
    print(f"[Done] Skipped images:   {skipped}")
    print(f"[Done] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
