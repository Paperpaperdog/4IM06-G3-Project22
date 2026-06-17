import argparse
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter, defaultdict


CLASS_MAP = {
    "original": "original_or_uncertain",
    "jpeg": "jpeg_compression",
    "JPEG_Q80": "jpeg_compression",
    "resample_x8": "8x8_resampling",
    "jpeg_then_resample_x8": "mixed",
    "resample_x8_then_jpeg": "mixed",
    "mix": "mixed",
    # Global up-sampling folders produced with --include_upsampling. The detector
    # has no native up-sampling output, so these rows measure how it responds to
    # (out-of-design) up-sampled inputs.
    "upsample_x2": "upsampling",
    "upsample_x4": "upsampling",
    "upsample_x8": "upsampling",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Detector decision thresholds (kept identical to jpeg_resample_detector.main defaults
# so that the in-process path reproduces the original per-image subprocess behaviour).
N_TESTS_PER_FEATURE = 8
THETA_JPEG = 3.0
THETA_RESAMPLE = 3.0
DELTA = 2.0


def load_detector_module(detector_path: str):
    """Import jpeg_resample_detector.py as a module from an arbitrary path.

    The file name contains no package and is imported by location so the
    evaluator works regardless of the current working directory.
    """
    detector_path = str(Path(detector_path).resolve())
    spec = importlib.util.spec_from_file_location("jpeg_resample_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load detector module from {detector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Normalized labels that mean "the image was geometrically resampled
# (up- or down-scaled)". Used to collapse the detector decision onto the common
# binary axis shared by the Mask/CNN methods in unified_method_comparison.py.
RESAMPLED_LABELS = {"8x8_resampling", "mixed", "upsampling"}


def is_resampling_label(label: str) -> bool:
    return label in RESAMPLED_LABELS


def normalize_prediction(pred: str) -> str:
    if pred == "jpeg_compression":
        return "jpeg_compression"

    if pred == "8x8_resampling":
        return "8x8_resampling"

    if pred in [
        "jpeg_and_8x8_resampling",
        "jpeg_dominant_possible_jpeg_plus_resampling",
        "resampling_dominant_possible_jpeg_plus_resampling",
    ]:
        return "mixed"

    if pred == "original_or_uncertain":
        return "original_or_uncertain"

    return pred


# ----------------------------------------------------------------------------
# Worker state (one detector module + shared null distribution per process).
# ----------------------------------------------------------------------------
_WORKER = {}


def _worker_init(detector_path: str, null_features, max_size: int):
    global _WORKER
    module = load_detector_module(detector_path)
    _WORKER = {
        "module": module,
        "null": null_features,
        "max_size": int(max_size),
    }


def _classify_one(task):
    """Classify a single image in-process. Returns (true_norm, pred_norm, name)."""
    img_path, true_norm = task
    module = _WORKER["module"]
    null_features = _WORKER["null"]
    max_size = _WORKER["max_size"]

    try:
        img = module.load_grayscale_image(img_path, max_size=max_size)
        img = module.crop_to_multiple_of_8(img)
        observed = module.extract_features(img)
        result = module.classify_features(
            observed_features=observed,
            null_features=null_features,
            n_tests_per_feature=N_TESTS_PER_FEATURE,
            theta_jpeg=THETA_JPEG,
            theta_resample=THETA_RESAMPLE,
            delta=DELTA,
        )
        pred_norm = normalize_prediction(result["label"])
    except Exception as exc:  # keep going, mirror the lenient subprocess behaviour
        pred_norm = "parse_failed"
        print(f"[Warning] Failed on {Path(img_path).name}: {exc}")

    return true_norm, pred_norm, Path(img_path).name


def collect_tasks(split_dir: Path, max_per_class: int):
    """Build the (image_path, true_norm) task list, preserving class ordering/logging."""
    tasks = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        true_class = cls_dir.name
        image_paths = [
            p for p in cls_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]
        image_paths = sorted(image_paths)[:max_per_class]

        print(f"[Info] Evaluating class {true_class}: {len(image_paths)} images")

        true_norm = CLASS_MAP.get(true_class, true_class)
        for p in image_paths:
            tasks.append((str(p), true_norm))

    return tasks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        help="Path to jpeg_resample_detector.py"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset_x8"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "none"],
        help="Use 'none' when class folders sit directly under --dataset_root "
             "(e.g. create_forensic_postprocess_dataset.py output)."
    )

    parser.add_argument(
        "--null_dir",
        type=str,
        required=True,
        help="Usually dataset_x8/train/original"
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=50,
        help="Maximum number of images evaluated per class"
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=512,
        help="Max image side length passed through to the detector (input-size study)."
    )

    parser.add_argument(
        "--null_max_images",
        type=int,
        default=100,
        help="Number of clean images used to build the (single, shared) null distribution."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. 0 = use all CPU cores, 1 = single process."
    )

    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to dump metrics (accuracy, confusion, binary "
             "resampling-detected accuracy) for unified_method_comparison.py."
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split_dir = dataset_root if args.split == "none" else dataset_root / args.split

    detector_path = str(Path(args.detector).resolve())

    # Build the empirical null distribution ONCE and reuse it for every image.
    # The original tool rebuilt it inside each per-image subprocess; building it
    # a single time is both far faster and more consistent across the dataset.
    print(f"[Info] Building shared null distribution from {args.null_dir} ...")
    base_module = load_detector_module(detector_path)
    null_features = base_module.build_null_features_from_directory(
        args.null_dir,
        max_images=args.null_max_images,
        max_size=args.max_size,
    )
    print(f"[Info] Number of null samples: {len(null_features)}")

    tasks = collect_tasks(split_dir, args.max_per_class)

    if args.workers == 0:
        workers = os.cpu_count() or 1
    else:
        workers = max(1, args.workers)

    y_true = []
    y_pred = []

    if workers == 1:
        _worker_init(detector_path, null_features, args.max_size)
        for task in tasks:
            true_norm, pred_norm, name = _classify_one(task)
            y_true.append(true_norm)
            y_pred.append(pred_norm)
            print(f"{name}: true={true_norm}, pred={pred_norm}")
    else:
        print(f"[Info] Running detector in-process across {workers} workers ...")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(detector_path, null_features, args.max_size),
        ) as pool:
            for true_norm, pred_norm, name in pool.map(_classify_one, tasks, chunksize=4):
                y_true.append(true_norm)
                y_pred.append(pred_norm)
                print(f"{name}: true={true_norm}, pred={pred_norm}")

    # Simple accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    acc = correct / total if total > 0 else 0

    # Common binary axis: "geometric resampling detected vs not" (original/JPEG
    # are negatives). Lets Route A sit on the same plot as Mask/CNN.
    binary_correct = sum(
        is_resampling_label(t) == is_resampling_label(p)
        for t, p in zip(y_true, y_pred)
    )
    binary_acc = binary_correct / total if total > 0 else 0

    print("\n================ Evaluation Result ================")
    print(f"Total:    {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"BinaryResamplingAccuracy: {binary_acc:.4f}")

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    matrix = defaultdict(Counter)

    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    print("\n================ Confusion Matrix ================")
    print("true \\ pred".ljust(35), end="")
    for lab in labels:
        print(lab[:18].ljust(20), end="")
    print()

    for t in labels:
        print(t[:32].ljust(35), end="")
        for p in labels:
            print(str(matrix[t][p]).ljust(20), end="")
        print()

    if args.json_out:
        out = {
            "total": total,
            "accuracy": acc,
            "binary_resampling_accuracy": binary_acc,
            "labels": labels,
            "confusion_matrix": [[matrix[t][p] for p in labels] for t in labels],
            "max_size": args.max_size,
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(f"Saved metrics JSON to {json_path}")


if __name__ == "__main__":
    main()
