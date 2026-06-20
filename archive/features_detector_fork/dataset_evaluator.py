import argparse
import importlib.util
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSES = ["original", "jpeg", "upsample_x8"]

LABEL_MAP = {
    "original_or_uncertain": "original",
    "jpeg_compression": "jpeg",
    "upsample_x8": "upsample_x8",
}


def progress(iterable, desc):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, unit="img")
    return iterable


def load_detector(path: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("detector_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load detector: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_files(root: Path):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def collect_samples(dataset_root: Path, max_per_class: int):
    samples = []

    for cls in CLASSES:
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            print(f"[Warning] Missing class folder: {cls_dir}")
            continue

        paths = image_files(cls_dir)[:max_per_class]
        samples.extend((p, cls) for p in paths)

    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_root}")

    return samples


def extract_one(detector, path: Path, max_size: int):
    img = detector.load_grayscale_image(str(path), max_size=max_size)
    img = detector.crop_to_multiple_of_8(img)
    return detector.extract_features(img)


def build_null_features(detector, null_dir: Path, max_null_images: int, max_size: int):
    paths = image_files(null_dir)[:max_null_images]

    if not paths:
        raise RuntimeError(f"No valid null images found in {null_dir}")

    null_features = []
    for path in progress(paths, "Building null"):
        try:
            null_features.append(extract_one(detector, path, max_size))
        except Exception as e:
            print(f"[Warning] Failed null image {path.name}: {e}")

    if not null_features:
        raise RuntimeError("All null images failed.")

    return null_features


def predict(detector, path: Path, null_features, args):
    features = extract_one(detector, path, args.max_size)
    result = detector.classify_features(
        observed_features=features,
        null_features=null_features,
        n_tests_per_feature=args.n_tests_per_feature,
        theta_jpeg=args.theta_jpeg,
        theta_resample=args.theta_resample,
        delta=args.delta,
    )

    pred = LABEL_MAP.get(result["label"], result["label"])
    return pred, result


def build_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    label_to_idx = {label: i for i, label in enumerate(CLASSES)}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    return matrix


def print_confusion_matrix(matrix):
    print("\n================ Confusion Matrix ================")
    print("true \\ pred".ljust(16), end="")
    for label in CLASSES:
        print(label.ljust(16), end="")
    print()

    for i, true_label in enumerate(CLASSES):
        print(true_label.ljust(16), end="")
        for j in range(len(CLASSES)):
            print(str(matrix[i, j]).ljust(16), end="")
        print()


def save_confusion_heatmap(matrix, output_path: str, normalize: bool = False):
    data = matrix.astype(np.float32)

    if normalize:
        row_sum = data.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        data = data / row_sum
        title = "Normalized Confusion Matrix"
        value_format = ".2f"
    else:
        title = "Confusion Matrix"
        value_format = ".0f"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)

    threshold = data.max() / 2.0 if data.size else 0.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                format(data[i, j], value_format),
                ha="center",
                va="center",
                color="white" if data[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Clean evaluator for original / JPEG / bicubic upsample x8 with confusion-matrix heatmap."
    )

    parser.add_argument("--detector", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--null_dir", required=True)

    parser.add_argument("--max_per_class", type=int, default=100000)
    parser.add_argument("--max_null_images", type=int, default=30)
    parser.add_argument("--max_size", type=int, default=1024)

    parser.add_argument("--theta_jpeg", type=float, default=3.0)
    parser.add_argument("--theta_resample", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--n_tests_per_feature", type=int, default=8)

    parser.add_argument("--print_each", action="store_true")
    parser.add_argument(
        "--heatmap_path",
        type=str,
        default="confusion_matrix_heatmap.png",
        help="Path for the raw-count confusion-matrix heatmap."
    )
    parser.add_argument(
        "--normalized_heatmap_path",
        type=str,
        default="confusion_matrix_heatmap_normalized.png",
        help="Path for the row-normalized confusion-matrix heatmap."
    )

    args = parser.parse_args()

    detector = load_detector(args.detector)
    dataset_root = Path(args.dataset_root)
    null_dir = Path(args.null_dir)

    samples = collect_samples(dataset_root, args.max_per_class)
    counts = Counter(label for _, label in samples)

    print(f"[Info] Dataset root: {dataset_root}")
    print(f"[Info] Total images: {len(samples)}")
    for cls in CLASSES:
        print(f"[Info] Class {cls}: {counts[cls]} images")

    null_features = build_null_features(
        detector=detector,
        null_dir=null_dir,
        max_null_images=args.max_null_images,
        max_size=args.max_size,
    )

    y_true, y_pred = [], []
    failures = 0

    for path, true_label in progress(samples, "Evaluating"):
        try:
            pred_label, result = predict(detector, path, null_features, args)

            y_true.append(true_label)
            y_pred.append(pred_label)

            if args.print_each:
                print(
                    f"{path.name}: true={true_label}, pred={pred_label}, "
                    f"jpeg={result['jpeg_score']:.4f}, "
                    f"resample={result['resample_score']:.4f}, "
                    f"up={result.get('upsample_score', 0.0):.4f}"
                )

        except Exception as e:
            failures += 1
            print(f"[Warning] Failed {path}: {e}")

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0

    print("\n================ Evaluation Result ================")
    print(f"Evaluated: {total}")
    print(f"Failures:  {failures}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.4f}")

    matrix = build_confusion_matrix(y_true, y_pred)
    print_confusion_matrix(matrix)

    save_confusion_heatmap(matrix, args.heatmap_path, normalize=False)
    save_confusion_heatmap(matrix, args.normalized_heatmap_path, normalize=True)

    print(f"\n[Done] Heatmap saved to: {args.heatmap_path}")
    print(f"[Done] Normalized heatmap saved to: {args.normalized_heatmap_path}")


if __name__ == "__main__":
    main()
