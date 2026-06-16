import argparse
import io
from pathlib import Path
from typing import List

from PIL import Image


# ============================================================
# Dataset post-processing script
# ============================================================
# This script does NOT train a network.
# It only creates forensic post-processing versions of a folder of images:
#
# output_dir/
#     original/      optional
#     jpeg/
#     resample_x8/
#     mix/
#
# The generated files are saved as PNG. JPEG samples are JPEG-compressed
# internally and then decoded back to PNG, so the final saving step does not
# introduce an additional JPEG compression.
# ============================================================


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESAMPLE_PERIOD = 8


def list_images(input_dir: str) -> List[Path]:
    input_dir = Path(input_dir)
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_to_multiple_of_8(img: Image.Image) -> Image.Image:
    """
    Crop the image to multiples of 8.

    This avoids incomplete 8x8 blocks when applying x8 block-wise resampling.
    It also makes all generated versions of the same image spatially aligned.
    """
    w, h = img.size
    w8 = w - w % RESAMPLE_PERIOD
    h8 = h - h % RESAMPLE_PERIOD

    if w8 < RESAMPLE_PERIOD or h8 < RESAMPLE_PERIOD:
        raise ValueError(f"Image too small after crop: {w}x{h}")

    return img.crop((0, 0, w8, h8))


def save_png(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def jpeg_compress_decode(img: Image.Image, quality: int) -> Image.Image:
    """
    Apply JPEG compression and decode back to RGB.

    The returned image should be saved as PNG to preserve the JPEG artifact
    without adding another JPEG compression at writing time.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def blockwise_x8_resample(
    img: Image.Image,
    inner_delta: int = -1,
    interpolation: str = "bicubic"
) -> Image.Image:
    """
    Create x8 block-wise resampling artifacts.

    inner_delta = -1:
        8x8 -> 7x7 -> 8x8

    inner_delta = +1:
        8x8 -> 9x9 -> 8x8

    This generates period-8 interpolation traces that may be confused with
    JPEG 8x8 block artifacts in the spectrum.
    """
    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }

    if interpolation not in interp_map:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    interp = interp_map[interpolation]
    img = crop_to_multiple_of_8(img.convert("RGB"))

    w, h = img.size
    out = Image.new("RGB", (w, h))

    inner = RESAMPLE_PERIOD + inner_delta
    inner = max(2, inner)

    for y in range(0, h, RESAMPLE_PERIOD):
        for x in range(0, w, RESAMPLE_PERIOD):
            block = img.crop((x, y, x + RESAMPLE_PERIOD, y + RESAMPLE_PERIOD))

            tmp = block.resize((inner, inner), interp)
            restored = tmp.resize((RESAMPLE_PERIOD, RESAMPLE_PERIOD), interp)

            out.paste(restored, (x, y))

    return out


def make_versions(
    img: Image.Image,
    quality: int,
    interpolation: str,
    inner_delta: int,
    mix_order: str
) -> dict:
    """
    Generate requested post-processing versions.

    mix_order:
        jpeg_then_resample:
            I -> JPEG(I) -> R8(JPEG(I))

        resample_then_jpeg:
            I -> R8(I) -> JPEG(R8(I))

        both:
            generate both mixed orders into the mix folder.
    """
    img = crop_to_multiple_of_8(img)

    versions = {}

    versions["original"] = img
    versions["jpeg"] = jpeg_compress_decode(img, quality=quality)
    versions["resample_x8"] = blockwise_x8_resample(
        img,
        inner_delta=inner_delta,
        interpolation=interpolation
    )

    if mix_order in {"jpeg_then_resample", "both"}:
        jpeg_first = jpeg_compress_decode(img, quality=quality)
        versions["mix_jpeg_then_resample_x8"] = blockwise_x8_resample(
            jpeg_first,
            inner_delta=inner_delta,
            interpolation=interpolation
        )

    if mix_order in {"resample_then_jpeg", "both"}:
        resample_first = blockwise_x8_resample(
            img,
            inner_delta=inner_delta,
            interpolation=interpolation
        )
        versions["mix_resample_x8_then_jpeg"] = jpeg_compress_decode(
            resample_first,
            quality=quality
        )

    return versions


def main():
    parser = argparse.ArgumentParser(
        description="Create JPEG, x8 resample, and mixed post-processing versions of an image dataset."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input folder containing PNG/images with arbitrary sizes."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder for generated dataset."
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality used for JPEG versions."
    )

    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation method used for x8 block-wise resampling."
    )

    parser.add_argument(
        "--inner_delta",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="Use -1 for 8x8->7x7->8x8, or +1 for 8x8->9x9->8x8."
    )

    parser.add_argument(
        "--mix_order",
        type=str,
        default="jpeg_then_resample",
        choices=["jpeg_then_resample", "resample_then_jpeg", "both"],
        help="Which mixed processing order to generate."
    )

    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Also copy/crop original images into output_dir/original."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    paths = list_images(input_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No valid images found in {input_dir}")

    print(f"[Info] Found {len(paths)} images.")
    print(f"[Info] Fixed resampling period: x8")
    print(f"[Info] JPEG quality: {args.quality}")
    print(f"[Info] Interpolation: {args.interpolation}")
    print(f"[Info] inner_delta: {args.inner_delta}")
    print(f"[Info] mix_order: {args.mix_order}")

    if args.include_original:
        (output_dir / "original").mkdir(parents=True, exist_ok=True)

    (output_dir / "jpeg").mkdir(parents=True, exist_ok=True)
    (output_dir / "resample_x8").mkdir(parents=True, exist_ok=True)
    (output_dir / "mix").mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    delta_tag = "m1" if args.inner_delta == -1 else "p1"

    for idx, path in enumerate(paths):
        try:
            img = open_rgb(path)
            versions = make_versions(
                img=img,
                quality=args.quality,
                interpolation=args.interpolation,
                inner_delta=args.inner_delta,
                mix_order=args.mix_order
            )

            stem = path.stem

            if args.include_original:
                out_name = f"{stem}_original_p8.png"
                save_png(versions["original"], output_dir / "original" / out_name)

            jpeg_name = f"{stem}_jpeg_q{args.quality}.png"
            save_png(versions["jpeg"], output_dir / "jpeg" / jpeg_name)

            resample_name = (
                f"{stem}_resample_x8_d{delta_tag}_{args.interpolation}.png"
            )
            save_png(versions["resample_x8"], output_dir / "resample_x8" / resample_name)

            if "mix_jpeg_then_resample_x8" in versions:
                mix_name = (
                    f"{stem}_mix_jpeg_then_resample_x8"
                    f"_q{args.quality}_d{delta_tag}_{args.interpolation}.png"
                )
                save_png(
                    versions["mix_jpeg_then_resample_x8"],
                    output_dir / "mix" / mix_name
                )

            if "mix_resample_x8_then_jpeg" in versions:
                mix_name = (
                    f"{stem}_mix_resample_x8_then_jpeg"
                    f"_q{args.quality}_d{delta_tag}_{args.interpolation}.png"
                )
                save_png(
                    versions["mix_resample_x8_then_jpeg"],
                    output_dir / "mix" / mix_name
                )

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
