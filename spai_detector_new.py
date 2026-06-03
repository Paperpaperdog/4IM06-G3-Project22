import os
import cv2
import csv
import glob
import joblib
import random
import argparse
import numpy as np

from tqdm import tqdm
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# Class labels
# ============================================================

CLASS_LABELS = [
    "original",
    "jpeg",
    "resampled",
    "jpeg_then_resampled",
    "resampled_then_jpeg"
]


# ============================================================
# Basic image loading and preprocessing
# ============================================================

def read_image(path):
    """
    Read image as RGB uint8.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fixed_center_crop(img, crop_size=256):
    """
    Center crop image to a fixed size without resizing.
    """
    h, w = img.shape[:2]

    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image is smaller than crop size {crop_size}: "
            f"image size = {w}x{h}"
        )

    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2

    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def fixed_random_crop(img, crop_size=256):
    """
    Random crop image to a fixed size without resizing.

    Used during training to increase patch diversity without introducing
    artificial resampling artifacts.
    """
    h, w = img.shape[:2]

    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image is smaller than crop size {crop_size}: "
            f"image size = {w}x{h}"
        )

    y0 = random.randint(0, h - crop_size)
    x0 = random.randint(0, w - crop_size)

    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def preprocess_image(img, image_size=256, random_crop=False):
    """
    Convert image to fixed size RGB by cropping only.

    No resizing is used here.
    """
    if random_crop:
        img = fixed_random_crop(img, crop_size=image_size)
    else:
        img = fixed_center_crop(img, crop_size=image_size)

    return img


def to_gray_float(img):
    """
    RGB uint8 -> grayscale float32 in [0, 1].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    return gray


# ============================================================
# Simulated post-processing operations
# ============================================================

def jpeg_compress(img, quality=80):
    """
    Simulate JPEG compression.
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    success, enc = cv2.imencode(".jpg", img_bgr, encode_param)
    if not success:
        raise RuntimeError("JPEG encoding failed.")

    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return dec


def resample_image(img, scale=0.75):
    """
    Simulate resampling and return to original size.
    """
    h, w = img.shape[:2]

    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))

    if scale < 1.0:
        interp1 = cv2.INTER_AREA
    else:
        interp1 = cv2.INTER_CUBIC

    small_or_large = cv2.resize(img, (new_w, new_h), interpolation=interp1)
    restored = cv2.resize(small_or_large, (w, h), interpolation=cv2.INTER_CUBIC)

    return restored


def jpeg_then_resample(img):
    """
    JPEG compression first, then resampling.
    """
    q = random.choice([70, 80, 90])
    s = random.choice([0.5, 0.75, 1.25, 1.5])

    out = jpeg_compress(img, quality=q)
    out = resample_image(out, scale=s)
    return out


def resample_then_jpeg(img):
    """
    Resampling first, then JPEG compression.
    """
    q = random.choice([70, 80, 90])
    s = random.choice([0.5, 0.75, 1.25, 1.5])

    out = resample_image(img, scale=s)
    out = jpeg_compress(out, quality=q)
    return out


def make_variant(img, label):
    """
    Generate one processed image according to label.
    """
    if label == "original":
        return img.copy()

    elif label == "jpeg":
        q = random.choice([60, 70, 80, 90, 95])
        return jpeg_compress(img, quality=q)

    elif label == "resampled":
        s = random.choice([0.5, 0.65, 0.75, 1.25, 1.5, 1.75])
        return resample_image(img, scale=s)

    elif label == "jpeg_then_resampled":
        return jpeg_then_resample(img)

    elif label == "resampled_then_jpeg":
        return resample_then_jpeg(img)

    else:
        raise ValueError(f"Unknown label: {label}")


# ============================================================
# Spectral feature extraction
# ============================================================

def radial_profile(log_spectrum, num_bins=64):
    """
    Compute radial average of 2D Fourier log magnitude.
    """
    h, w = log_spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r = r / r.max()
    bins = np.linspace(0, 1, num_bins + 1)

    features = []
    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            features.append(log_spectrum[mask].mean())
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def normalized_log_fft(gray):
    """
    Compute normalized log FFT magnitude.
    """
    gray = gray.astype(np.float32)
    gray = gray - gray.mean()

    spectrum = fftshift(fft2(gray))
    magnitude = np.abs(spectrum)
    log_mag = np.log1p(magnitude)

    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)
    return log_mag.astype(np.float32)


def fft_features(gray, num_bins=64):
    """
    Extract global FFT radial profile features.
    """
    log_mag = normalized_log_fft(gray)

    radial = radial_profile(log_mag, num_bins=num_bins)

    stats = np.array([
        log_mag.mean(),
        log_mag.std(),
        np.percentile(log_mag, 25),
        np.percentile(log_mag, 50),
        np.percentile(log_mag, 75),
        log_mag.max()
    ], dtype=np.float32)

    return np.concatenate([radial, stats])


def residual_fft_features(gray, num_bins=64):
    """
    Extract FFT features from high-pass residual.
    """
    residual = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return fft_features(residual, num_bins=num_bins)


def summarize_directional_peaks(profile, ignore_low_freq=4):
    """
    Summarize peaks in a 1D directional FFT profile.
    """
    n = len(profile)
    center = n // 2
    max_dist = min(center, n - center - 1)

    values = []
    for d in range(1, max_dist + 1):
        values.append((profile[center - d] + profile[center + d]) / 2.0)

    values = np.array(values, dtype=np.float32)

    if len(values) <= ignore_low_freq + 2:
        return np.zeros(8, dtype=np.float32)

    valid = values[ignore_low_freq:]

    threshold = valid.mean() + 0.5 * valid.std()
    peaks, props = find_peaks(valid, height=threshold, distance=2)

    if len(peaks) == 0:
        return np.array([
            0.0,
            0.0,
            0.0,
            valid.mean(),
            valid.std(),
            valid.max(),
            0.0,
            0.0
        ], dtype=np.float32)

    heights = props["peak_heights"]
    top_idx = np.argmax(heights)

    top_position = peaks[top_idx] + ignore_low_freq + 1
    top_height = heights[top_idx]

    sorted_heights = np.sort(heights)[::-1]
    top3_sum = sorted_heights[:3].sum()

    peak_ratio = top_height / (valid.mean() + 1e-8)
    peak_density = len(peaks) / (len(valid) + 1e-8)

    return np.array([
        len(peaks),
        top_position,
        top_height,
        heights.mean(),
        heights.std(),
        top3_sum,
        peak_ratio,
        peak_density
    ], dtype=np.float32)


def directional_fft_peak_features(gray):
    """
    Extract 2D FFT directional peak features.
    """
    residual = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    log_mag = normalized_log_fft(residual)

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2

    horizontal_profile = log_mag[cy, :]
    vertical_profile = log_mag[:, cx]

    main_diag_profile = np.diag(log_mag)
    anti_diag_profile = np.diag(np.fliplr(log_mag))

    f_horizontal = summarize_directional_peaks(horizontal_profile)
    f_vertical = summarize_directional_peaks(vertical_profile)
    f_main_diag = summarize_directional_peaks(main_diag_profile)
    f_anti_diag = summarize_directional_peaks(anti_diag_profile)

    return np.concatenate([
        f_horizontal,
        f_vertical,
        f_main_diag,
        f_anti_diag
    ])


def jpeg_block_features(gray):
    """
    Extract simple JPEG 8x8 block artifact features.
    """
    h, w = gray.shape

    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    vertical_boundary_cols = np.arange(7, w - 1, 8)
    horizontal_boundary_rows = np.arange(7, h - 1, 8)

    if len(vertical_boundary_cols) > 0:
        boundary_v = dx[:, vertical_boundary_cols].mean()
    else:
        boundary_v = 0.0

    if len(horizontal_boundary_rows) > 0:
        boundary_h = dy[horizontal_boundary_rows, :].mean()
    else:
        boundary_h = 0.0

    non_boundary_cols = np.setdiff1d(np.arange(w - 1), vertical_boundary_cols)
    non_boundary_rows = np.setdiff1d(np.arange(h - 1), horizontal_boundary_rows)

    non_boundary_v = dx[:, non_boundary_cols].mean()
    non_boundary_h = dy[non_boundary_rows, :].mean()

    ratio_v = boundary_v / (non_boundary_v + 1e-8)
    ratio_h = boundary_h / (non_boundary_h + 1e-8)

    features = np.array([
        boundary_v,
        boundary_h,
        non_boundary_v,
        non_boundary_h,
        ratio_v,
        ratio_h,
        max(ratio_v, ratio_h),
        (ratio_v + ratio_h) / 2.0
    ], dtype=np.float32)

    return features


def dct_coefficient_features(gray):
    """
    Extract 8x8 block DCT coefficient features.
    """
    gray_255 = gray.astype(np.float32) * 255.0

    h, w = gray_255.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    gray_255 = gray_255[:h8, :w8]

    dct_blocks = []

    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray_255[y:y + 8, x:x + 8]
            block = block - 128.0
            coeff = cv2.dct(block)
            dct_blocks.append(coeff)

    if len(dct_blocks) == 0:
        return np.zeros(78, dtype=np.float32)

    dct_blocks = np.stack(dct_blocks, axis=0)

    abs_coeff = np.abs(dct_blocks)

    mean_abs_coeff = abs_coeff.mean(axis=0).reshape(-1)
    mean_abs_coeff = np.log1p(mean_abs_coeff)

    dc = dct_blocks[:, 0, 0]
    ac = dct_blocks.reshape(dct_blocks.shape[0], -1)[:, 1:]
    abs_ac = np.abs(ac)

    yy, xx = np.indices((8, 8))
    freq_sum = yy + xx
    flat_freq_sum = freq_sum.reshape(-1)

    low_mask = (flat_freq_sum >= 1) & (flat_freq_sum <= 2)
    mid_mask = (flat_freq_sum >= 3) & (flat_freq_sum <= 5)
    high_mask = flat_freq_sum >= 6

    flat_abs = abs_coeff.reshape(abs_coeff.shape[0], -1)

    low_energy = flat_abs[:, low_mask].mean()
    mid_energy = flat_abs[:, mid_mask].mean()
    high_energy = flat_abs[:, high_mask].mean()

    total_ac_energy = abs_ac.mean() + 1e-8

    small_ratio_1 = np.mean(abs_ac < 1.0)
    small_ratio_2 = np.mean(abs_ac < 2.0)
    small_ratio_5 = np.mean(abs_ac < 5.0)

    band_features = np.array([
        dc.mean(),
        dc.std(),
        low_energy,
        mid_energy,
        high_energy,
        low_energy / total_ac_energy,
        mid_energy / total_ac_energy,
        high_energy / total_ac_energy,
        high_energy / (low_energy + 1e-8),
        high_energy / (mid_energy + 1e-8),
        small_ratio_1,
        small_ratio_2,
        small_ratio_5,
        total_ac_energy
    ], dtype=np.float32)

    return np.concatenate([
        mean_abs_coeff.astype(np.float32),
        band_features
    ])


def autocorrelation_1d(signal):
    """
    Normalized autocorrelation using FFT.
    """
    signal = signal.astype(np.float32)
    signal = signal - signal.mean()

    n = len(signal)
    f = np.fft.fft(signal, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]

    ac = ac / (ac[0] + 1e-8)
    return ac


def resampling_periodicity_features(gray):
    """
    Extract periodicity features related to resampling.
    """
    residual = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    horizontal_signal = np.mean(np.abs(residual), axis=0)
    vertical_signal = np.mean(np.abs(residual), axis=1)

    ac_x = autocorrelation_1d(horizontal_signal)
    ac_y = autocorrelation_1d(vertical_signal)

    ac_x_valid = ac_x[2:80]
    ac_y_valid = ac_y[2:80]

    peaks_x, props_x = find_peaks(ac_x_valid, height=0.02)
    peaks_y, props_y = find_peaks(ac_y_valid, height=0.02)

    def summarize_peaks(peaks, props):
        if len(peaks) == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)

        heights = props["peak_heights"]

        top_idx = np.argmax(heights)
        top_peak_position = peaks[top_idx] + 2
        top_peak_height = heights[top_idx]

        return np.array([
            len(peaks),
            top_peak_position,
            top_peak_height,
            heights.mean(),
            heights.std()
        ], dtype=np.float32)

    feat_x = summarize_peaks(peaks_x, props_x)
    feat_y = summarize_peaks(peaks_y, props_y)

    return np.concatenate([feat_x, feat_y])


def extract_features(img, image_size=256, random_crop=False):
    """
    Extract SPAI-inspired spectral + JPEG + resampling features.

    random_crop=True is used during training.
    random_crop=False is used during prediction.
    """
    img = preprocess_image(
        img,
        image_size=image_size,
        random_crop=random_crop
    )

    gray = to_gray_float(img)

    f_fft = fft_features(gray, num_bins=64)
    f_residual_fft = residual_fft_features(gray, num_bins=64)
    f_directional_fft = directional_fft_peak_features(gray)
    f_jpeg = jpeg_block_features(gray)
    f_dct = dct_coefficient_features(gray)
    f_resample = resampling_periodicity_features(gray)

    features = np.concatenate([
        f_fft,
        f_residual_fft,
        f_directional_fft,
        f_jpeg,
        f_dct,
        f_resample
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


# ============================================================
# Dataset preparation
# ============================================================

def collect_image_paths(data_dir):
    """
    Collect common image files.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]

    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    paths = sorted(paths)
    return paths


def build_training_data(data_dir, image_size=256, variants_per_image=1, max_images=None):
    """
    Build training data from real image dataset.

    For each real image, generate:
    - original
    - jpeg
    - resampled
    - jpeg_then_resampled
    - resampled_then_jpeg

    Random crop is applied before generating each variant.
    """
    paths = collect_image_paths(data_dir)

    if max_images is not None:
        paths = paths[:max_images]

    labels = CLASS_LABELS

    X = []
    y = []
    meta = []

    for path in tqdm(paths, desc="Building dataset"):
        try:
            img = read_image(path)

            for label in labels:
                for _ in range(variants_per_image):
                    cropped_img = preprocess_image(
                        img,
                        image_size=image_size,
                        random_crop=True
                    )

                    variant = make_variant(cropped_img, label)

                    feat = extract_features(
                        variant,
                        image_size=image_size,
                        random_crop=False
                    )

                    X.append(feat)
                    y.append(label)
                    meta.append((path, label))

        except Exception as e:
            print(f"[Warning] Skip {path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y, meta


# ============================================================
# Training and evaluation
# ============================================================

def train_model(data_dir, output_model, image_size=256, variants_per_image=1, max_images=None):
    X, y, meta = build_training_data(
        data_dir=data_dir,
        image_size=image_size,
        variants_per_image=variants_per_image,
        max_images=max_images
    )

    print("Feature matrix:", X.shape)
    print("Labels:", sorted(set(y)))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    print("\nClassification Report:")
    print(classification_report(y_val, pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, pred, labels=CLASS_LABELS))

    package = {
        "model": model,
        "image_size": image_size,
        "classes": CLASS_LABELS
    }

    joblib.dump(package, output_model)
    print(f"\nSaved model to: {output_model}")


# ============================================================
# Prediction
# ============================================================

def predict_folder(model_path, input_dir, output_csv):
    package = joblib.load(model_path)

    model = package["model"]
    image_size = package["image_size"]

    paths = collect_image_paths(input_dir)

    rows = []

    for path in tqdm(paths, desc="Predicting"):
        try:
            img = read_image(path)

            feat = extract_features(
                img,
                image_size=image_size,
                random_crop=False
            ).reshape(1, -1)

            pred = model.predict(feat)[0]

            if hasattr(model.named_steps["clf"], "predict_proba"):
                probs = model.predict_proba(feat)[0]
                classes = model.named_steps["clf"].classes_

                prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            else:
                prob_dict = {}

            row = {
                "path": path,
                "prediction": pred,
                "prob_original": prob_dict.get("original", 0.0),
                "prob_jpeg": prob_dict.get("jpeg", 0.0),
                "prob_resampled": prob_dict.get("resampled", 0.0),
                "prob_jpeg_then_resampled": prob_dict.get("jpeg_then_resampled", 0.0),
                "prob_resampled_then_jpeg": prob_dict.get("resampled_then_jpeg", 0.0),
            }

            rows.append(row)

        except Exception as e:
            print(f"[Warning] Skip {path}: {e}")

    fieldnames = [
        "path",
        "prediction",
        "prob_original",
        "prob_jpeg",
        "prob_resampled",
        "prob_jpeg_then_resampled",
        "prob_resampled_then_jpeg"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {output_csv}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="train or predict"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="For train: real image dataset folder. For predict: test image folder."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="spai_style_detector.pkl",
        help="Path to save or load model."
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Prediction CSV path."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Fixed crop size used for spectral feature extraction."
    )

    parser.add_argument(
        "--variants_per_image",
        type=int,
        default=1,
        help="Number of generated variants per class for each real image."
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of real images for quick debugging."
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            data_dir=args.data_dir,
            output_model=args.model_path,
            image_size=args.image_size,
            variants_per_image=args.variants_per_image,
            max_images=args.max_images
        )

    elif args.mode == "predict":
        predict_folder(
            model_path=args.model_path,
            input_dir=args.data_dir,
            output_csv=args.output_csv
        )


if __name__ == "__main__":
    main()
