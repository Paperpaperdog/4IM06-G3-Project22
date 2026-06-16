import argparse
import subprocess
from pathlib import Path
from collections import Counter, defaultdict


CLASS_MAP = {
    "original": "original_or_uncertain",
    "jpeg": "jpeg_compression",
    "resample_x8": "8x8_resampling",
    "jpeg_then_resample_x8": "mixed",
    "resample_x8_then_jpeg": "mixed",
}


def parse_label(output_text: str) -> str:
    for line in output_text.splitlines():
        if line.startswith("Label:"):
            return line.replace("Label:", "").strip()
    return "parse_failed"


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
        choices=["train", "val", "test"]
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

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split_dir = dataset_root / args.split

    y_true = []
    y_pred = []

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        true_class = cls_dir.name
        image_paths = [
            p for p in cls_dir.rglob("*")
            if p.suffix.lower() in image_exts
        ]

        image_paths = image_paths[:args.max_per_class]

        print(f"[Info] Evaluating class {true_class}: {len(image_paths)} images")

        for img_path in image_paths:
            cmd = [
                "python",
                args.detector,
                "--image",
                str(img_path),
                "--null_dir",
                args.null_dir,
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            pred_raw = parse_label(result.stdout)
            pred_norm = normalize_prediction(pred_raw)

            true_norm = CLASS_MAP.get(true_class, true_class)

            y_true.append(true_norm)
            y_pred.append(pred_norm)

            print(f"{img_path.name}: true={true_norm}, pred={pred_norm}")

    # Simple accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    acc = correct / total if total > 0 else 0

    print("\n================ Evaluation Result ================")
    print(f"Total:    {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.4f}")

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


if __name__ == "__main__":
    main()