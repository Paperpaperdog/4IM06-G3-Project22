import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import convolve, median_filter, maximum_filter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# Utilities
# ============================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PERIOD = 8


def progress_iter(iterable, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, ncols=90)
    return iterable


def crop_to_multiple_of_8(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    h8 = h - h % PERIOD
    w8 = w - w % PERIOD
    if h8 < PERIOD or w8 < PERIOD:
        raise ValueError(f"Image too small after crop-to-multiple-of-8: {w}x{h}")
    return img[:h8, :w8]


def load_grayscale_image(path: str, max_size: int = 512) -> np.ndarray:
    """
    Load image as grayscale float32 in [0, 255].

    This version uses TOP-LEFT crop, not center crop.
    It does not resize the image, so the detector itself does not introduce
    extra interpolation artifacts.
    """
    img = Image.open(path).convert("L")
    w, h = img.size

    crop_w = min(w, max_size)
    crop_h = min(h, max_size)
    crop_w = crop_w - crop_w % PERIOD
    crop_h = crop_h - crop_h % PERIOD

    if crop_w < PERIOD or crop_h < PERIOD:
        raise ValueError(f"Image too small after crop-to-multiple-of-8: {w}x{h}")

    # Top-left crop.
    img = img.crop((0, 0, crop_w, crop_h))
    return np.asarray(img).astype(np.float32)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)


# ============================================================
# Basic maps
# ============================================================

def prediction_residual(img: np.ndarray) -> np.ndarray:
    """
    Simple prediction residual:
        r(x, y) = I(x, y) - mean of 4-neighborhood
    """
    kernel = np.array([
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0],
    ], dtype=np.float32)
    pred = convolve(img, kernel, mode="reflect")
    return img - pred


def second_difference_energy(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    dx2 = img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2]
    dy2 = img[2:, :] - 2 * img[1:-1, :] + img[:-2, :]
    return float(0.5 * (np.mean(np.abs(dx2)) + np.mean(np.abs(dy2))))


def second_difference_map(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    dx2 = np.zeros_like(img, dtype=np.float32)
    dy2 = np.zeros_like(img, dtype=np.float32)
    dx2[:, 1:-1] = np.abs(img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2])
    dy2[1:-1, :] = np.abs(img[2:, :] - 2 * img[1:-1, :] + img[:-2, :])
    return 0.5 * (dx2 + dy2)


def residual_autocorrelation_lag(residual: np.ndarray, lag: int) -> float:
    r = normalize_image(residual)
    vals = []
    if r.shape[1] > lag:
        vals.append(float(np.mean(r[:, :-lag] * r[:, lag:])))
    if r.shape[0] > lag:
        vals.append(float(np.mean(r[:-lag, :] * r[lag:, :])))
    return float(np.mean(vals)) if vals else 0.0


def phase8_statistics(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Split pixels by (row mod 8, col mod 8), then compute phase imbalance.
    """
    arr = np.abs(arr.astype(np.float32))
    h, w = arr.shape
    h = h - h % PERIOD
    w = w - w % PERIOD

    if h < PERIOD or w < PERIOD:
        return {
            f"{prefix}_range_ratio": 0.0,
            f"{prefix}_min_ratio": 1.0,
        }

    arr = arr[:h, :w]
    phase_means = []

    for yy in range(PERIOD):
        for xx in range(PERIOD):
            vals = arr[yy::PERIOD, xx::PERIOD]
            if vals.size > 0:
                phase_means.append(float(vals.mean()))

    phase_means = np.array(phase_means, dtype=np.float32)
    mean_val = float(phase_means.mean()) + 1e-8
    return {
        f"{prefix}_range_ratio": float((phase_means.max() - phase_means.min()) / mean_val),
        f"{prefix}_min_ratio": float(phase_means.min() / mean_val),
    }


# ============================================================
# JPEG features
# ============================================================

def block_boundary_ratio(img: np.ndarray) -> float:
    """
    Ratio between 8-pixel block-boundary discontinuity and non-boundary
    discontinuity. JPEG tends to increase this ratio.
    """
    img = img.astype(np.float32)
    h, w = img.shape

    if w > 9:
        vertical_diff = np.abs(img[:, 1:] - img[:, :-1])
        cols = np.arange(1, w)
        boundary_cols = cols % PERIOD == 0
        non_boundary_cols = ~boundary_cols
        b_v = vertical_diff[:, boundary_cols].mean() if boundary_cols.any() else 0.0
        nb_v = vertical_diff[:, non_boundary_cols].mean() if non_boundary_cols.any() else 1e-8
    else:
        b_v, nb_v = 0.0, 1e-8

    if h > 9:
        horizontal_diff = np.abs(img[1:, :] - img[:-1, :])
        rows = np.arange(1, h)
        boundary_rows = rows % PERIOD == 0
        non_boundary_rows = ~boundary_rows
        b_h = horizontal_diff[boundary_rows, :].mean() if boundary_rows.any() else 0.0
        nb_h = horizontal_diff[non_boundary_rows, :].mean() if non_boundary_rows.any() else 1e-8
    else:
        b_h, nb_h = 0.0, 1e-8

    boundary = 0.5 * (b_v + b_h)
    non_boundary = 0.5 * (nb_v + nb_h)
    return float(boundary / (non_boundary + 1e-8))


def block_dct_8x8(img: np.ndarray) -> np.ndarray:
    img = crop_to_multiple_of_8(img)
    h, w = img.shape
    blocks = img.reshape(h // PERIOD, PERIOD, w // PERIOD, PERIOD)
    blocks = blocks.transpose(0, 2, 1, 3)
    coeffs = dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    return coeffs.astype(np.float32)


def dct_high_frequency_zero_ratio(coeffs: np.ndarray, zero_threshold: float = 1.0) -> float:
    hf_mask = np.zeros((PERIOD, PERIOD), dtype=bool)
    for u in range(PERIOD):
        for v in range(PERIOD):
            if u + v >= PERIOD:
                hf_mask[u, v] = True

    hf_coeffs = coeffs[:, :, hf_mask]
    return float(np.mean(np.abs(hf_coeffs) < zero_threshold))


def dct_histogram_comb_score(coeffs: np.ndarray, bins: int = 80) -> float:
    selected_positions = [
        (0, 1), (1, 0), (1, 1), (0, 2), (2, 0),
        (2, 1), (1, 2), (3, 0), (0, 3),
    ]

    scores = []
    for u, v in selected_positions:
        c = coeffs[:, :, u, v].ravel()
        lo, hi = np.percentile(c, [1, 99])
        if hi <= lo + 1e-8:
            continue

        hist, _ = np.histogram(c, bins=bins, range=(lo, hi))
        hist = hist.astype(np.float32)
        if hist.mean() <= 1e-8:
            continue

        second_diff = np.diff(hist, n=2)
        scores.append(float(np.mean(np.abs(second_diff)) / (hist.mean() + 1e-8)))

    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# Bicubic upsample x8 features
# ============================================================

def bicubic_x8_interpolation_features(img: np.ndarray) -> Dict[str, float]:
    """
    Features matched to bicubic upsample x8.

    Bicubic upsampling does not make exact 8x8 repeated blocks. Instead it
    tends to produce interpolation smoothness, stronger short-lag correlation,
    phase-dependent statistics modulo 8, and reduced relative high-frequency
    energy.
    """
    img = crop_to_multiple_of_8(img.astype(np.float32))
    h, w = img.shape

    if h < PERIOD * 2 or w < PERIOD * 2:
        return {
            "bicubic_up_low_second_diff_energy": 0.0,
            "bicubic_up_residual_autocorr_lag1": 0.0,
            "bicubic_up_residual_autocorr_lag2": 0.0,
            "bicubic_up_residual_autocorr_lag4": 0.0,
            "bicubic_up_second_diff_phase8_range_ratio": 0.0,
            "bicubic_up_second_diff_phase8_min_ratio": 1.0,
            "bicubic_up_gradient_phase8_range_ratio": 0.0,
            "bicubic_up_high_freq_energy_ratio": 1.0,
        }

    residual = prediction_residual(img)
    d2 = second_difference_map(img)

    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    grad_mag = np.sqrt(gx * gx + gy * gy)

    second_phase = phase8_statistics(d2, prefix="bicubic_up_second_diff_phase8")
    grad_phase = phase8_statistics(grad_mag, prefix="bicubic_up_gradient_phase8")

    z = normalize_image(img)
    F = np.fft.fftshift(np.fft.fft2(z))
    power = np.abs(F) ** 2
    yy, xx = np.indices(power.shape)
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2) + 1e-8
    high_mask = rr > 0.35 * max_r
    high_freq_ratio = float(power[high_mask].sum() / (power.sum() + 1e-8))

    return {
        "bicubic_up_low_second_diff_energy": second_difference_energy(img),
        "bicubic_up_residual_autocorr_lag1": residual_autocorrelation_lag(residual, 1),
        "bicubic_up_residual_autocorr_lag2": residual_autocorrelation_lag(residual, 2),
        "bicubic_up_residual_autocorr_lag4": residual_autocorrelation_lag(residual, 4),
        "bicubic_up_second_diff_phase8_range_ratio": second_phase["bicubic_up_second_diff_phase8_range_ratio"],
        "bicubic_up_second_diff_phase8_min_ratio": second_phase["bicubic_up_second_diff_phase8_min_ratio"],
        "bicubic_up_gradient_phase8_range_ratio": grad_phase["bicubic_up_gradient_phase8_range_ratio"],
        "bicubic_up_high_freq_energy_ratio": high_freq_ratio,
    }


# ============================================================
# Generic periodic resampling features
# ============================================================

def prediction_error_phase8_features(residual: np.ndarray) -> Dict[str, float]:
    return phase8_statistics(np.abs(residual), prefix="prediction_error_phase8")


def second_diff_phase8_features(img: np.ndarray) -> Dict[str, float]:
    d2 = second_difference_map(img)
    return phase8_statistics(d2, prefix="second_diff_phase8")


def get_8_periodic_peak_positions(shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    h, w = shape
    cy, cx = h // 2, w // 2
    dx = max(1, w // PERIOD)
    dy = max(1, h // PERIOD)

    candidates = [
        (cy, cx + dx), (cy, cx - dx),
        (cy + dy, cx), (cy - dy, cx),
        (cy, cx + 2 * dx), (cy, cx - 2 * dx),
        (cy + 2 * dy, cx), (cy - 2 * dy, cx),
    ]

    return [(y, x) for y, x in candidates if 0 <= y < h and 0 <= x < w]


def non_jpeg_periodic_peak_features(S_norm: np.ndarray) -> Dict[str, float]:
    """
    Count significant FFT peaks after suppressing common JPEG period-8 positions.
    """
    S = S_norm.astype(np.float32)
    h, w = S.shape
    cy, cx = h // 2, w // 2
    exclude_radius = 6
    local_size = 5
    threshold_std = 2.5

    mask = np.ones_like(S, dtype=bool)

    y0 = max(0, cy - exclude_radius)
    y1 = min(h, cy + exclude_radius + 1)
    x0 = max(0, cx - exclude_radius)
    x1 = min(w, cx + exclude_radius + 1)
    mask[y0:y1, x0:x1] = False

    for y, x in get_8_periodic_peak_positions(S.shape):
        y0 = max(0, y - exclude_radius)
        y1 = min(h, y + exclude_radius + 1)
        x0 = max(0, x - exclude_radius)
        x1 = min(w, x + exclude_radius + 1)
        mask[y0:y1, x0:x1] = False

    valid = S[mask]
    if valid.size == 0:
        return {
            "non_jpeg_periodic_peak_count": 0.0,
            "non_jpeg_periodic_peak_strength": 0.0,
        }

    threshold = float(valid.mean() + threshold_std * (valid.std() + 1e-8))
    local_max = S == maximum_filter(S, size=local_size, mode="reflect")
    peak_mask = mask & local_max & (S > threshold)

    peak_values = S[peak_mask]
    peak_count = int(peak_values.size)
    norm_area = max(float(h * w) / 10000.0, 1.0)
    normalized_count = float(peak_count / norm_area)
    strength = float(np.mean(peak_values - threshold)) if peak_count > 0 else 0.0

    return {
        "non_jpeg_periodic_peak_count": normalized_count,
        "non_jpeg_periodic_peak_strength": strength,
    }


def log_fft_spectrum(img: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F)).astype(np.float32)


def locally_normalize_spectrum(S: np.ndarray, size: int = 31) -> np.ndarray:
    bg = median_filter(S, size=size, mode="reflect")
    return S - bg


# ============================================================
# Feature extraction
# ============================================================

def extract_features(img: np.ndarray, show_progress: bool = False) -> Dict[str, float]:
    steps = progress_iter(range(5), desc="Feature extraction") if show_progress else range(5)
    step_iter = iter(steps)

    next(step_iter)
    img = crop_to_multiple_of_8(img)

    next(step_iter)
    residual = prediction_residual(img)

    next(step_iter)
    S_norm = locally_normalize_spectrum(log_fft_spectrum(residual))

    next(step_iter)
    coeffs = block_dct_8x8(img)

    next(step_iter)
    features = {
        # JPEG features.
        "block_boundary_ratio": block_boundary_ratio(img),
        "dct_hf_zero_ratio": dct_high_frequency_zero_ratio(coeffs),
        "dct_comb_score": dct_histogram_comb_score(coeffs),

        # Generic resampling features.
        "residual_autocorr_lag1": residual_autocorrelation_lag(residual, 1),
        "residual_autocorr_lag2": residual_autocorrelation_lag(residual, 2),
        "residual_autocorr_lag4": residual_autocorrelation_lag(residual, 4),
        "second_diff_energy": second_difference_energy(img),
    }

    features.update(prediction_error_phase8_features(residual))
    features.update(second_diff_phase8_features(img))
    features.update(non_jpeg_periodic_peak_features(S_norm))
    features.update(bicubic_x8_interpolation_features(img))

    return features


# ============================================================
# Null hypothesis and a-contrario scoring
# ============================================================

def phase_randomized_surrogate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std() + 1e-8

    F = np.fft.fft2(img)
    magnitude = np.abs(F)
    random_phase = rng.uniform(-np.pi, np.pi, size=img.shape)
    F_random = magnitude * np.exp(1j * random_phase)
    surrogate = np.real(np.fft.ifft2(F_random))

    surrogate = (surrogate - surrogate.mean()) / (surrogate.std() + 1e-8)
    surrogate = surrogate * std + mean
    return np.clip(surrogate, 0, 255).astype(np.float32)


def build_null_features_from_surrogates(
    img: np.ndarray,
    num_surrogates: int = 64,
    seed: int = 0
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    null_features = []
    for _ in progress_iter(range(num_surrogates), desc="Surrogate null samples"):
        surr = phase_randomized_surrogate(img, rng)
        null_features.append(extract_features(surr, show_progress=False))
    return null_features


def build_null_features_from_directory(
    null_dir: str,
    max_images: int = 30,
    max_size: int = 512
) -> List[Dict[str, float]]:
    paths = [
        p for p in Path(null_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    paths = sorted(paths)[:max_images]

    null_features = []
    for p in progress_iter(paths, desc="Directory null samples"):
        try:
            img = load_grayscale_image(str(p), max_size=max_size)
            null_features.append(extract_features(img, show_progress=False))
        except Exception as e:
            print(f"[Warning] Failed to process null image {p}: {e}")

    if len(null_features) == 0:
        raise ValueError("No valid null images found in null_dir.")

    return null_features


def empirical_p_value(observed: float, null_values: np.ndarray, tail: str) -> float:
    n = len(null_values)
    if tail == "upper":
        count = np.sum(null_values >= observed)
    elif tail == "lower":
        count = np.sum(null_values <= observed)
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
    return float((count + 1) / (n + 1))


def compute_nfa(observed: float, null_values: np.ndarray, tail: str, n_tests: int) -> Tuple[float, float]:
    p = empirical_p_value(observed, null_values, tail)
    return p, float(n_tests * p)


def nfa_to_score(nfa: float) -> float:
    nfa = max(float(nfa), 1e-300)
    return 0.0 if nfa >= 1.0 else float(-np.log10(nfa))


# ============================================================
# Classification
# ============================================================

JPEG_FEATURES = {
    "block_boundary_ratio": {"tail": "upper", "weight": 1.6, "description": "8x8 block boundary discontinuity"},
    "dct_hf_zero_ratio": {"tail": "upper", "weight": 1.8, "description": "high-frequency DCT zero ratio"},
    "dct_comb_score": {"tail": "upper", "weight": 1.6, "description": "DCT histogram comb-like quantization pattern"},
}

GENERIC_RESAMPLE_FEATURES = {
    "residual_autocorr_lag1": {"tail": "upper", "weight": 0.3, "description": "short-range residual autocorrelation"},
    "residual_autocorr_lag2": {"tail": "upper", "weight": 0.4, "description": "lag-2 residual autocorrelation"},
    "residual_autocorr_lag4": {"tail": "upper", "weight": 0.6, "description": "lag-4 residual autocorrelation"},
    "second_diff_energy": {"tail": "lower", "weight": 1.2, "description": "low second-difference energy"},
    "prediction_error_phase8_range_ratio": {"tail": "upper", "weight": 1.8, "description": "phase-8 prediction-error range"},
    "prediction_error_phase8_min_ratio": {"tail": "lower", "weight": 1.2, "description": "unusually smooth prediction-error phase"},
    "second_diff_phase8_range_ratio": {"tail": "upper", "weight": 1.8, "description": "phase-8 second-difference range"},
    "second_diff_phase8_min_ratio": {"tail": "lower", "weight": 1.2, "description": "unusually smooth second-difference phase"},
    "non_jpeg_periodic_peak_count": {"tail": "upper", "weight": 1.2, "description": "periodic FFT peaks outside JPEG grid"},
    "non_jpeg_periodic_peak_strength": {"tail": "upper", "weight": 1.0, "description": "strength of non-JPEG periodic FFT peaks"},
}

UPSAMPLE_X8_FEATURES = {
    "bicubic_up_low_second_diff_energy": {"tail": "lower", "weight": 1.4, "description": "bicubic interpolation smoothness"},
    "bicubic_up_residual_autocorr_lag1": {"tail": "upper", "weight": 0.8, "description": "bicubic lag-1 residual correlation"},
    "bicubic_up_residual_autocorr_lag2": {"tail": "upper", "weight": 0.7, "description": "bicubic lag-2 residual correlation"},
    "bicubic_up_residual_autocorr_lag4": {"tail": "upper", "weight": 0.6, "description": "bicubic lag-4 residual correlation"},
    "bicubic_up_second_diff_phase8_range_ratio": {"tail": "upper", "weight": 1.6, "description": "period-8 second-difference imbalance"},
    "bicubic_up_second_diff_phase8_min_ratio": {"tail": "lower", "weight": 1.2, "description": "smooth second-difference phase"},
    "bicubic_up_gradient_phase8_range_ratio": {"tail": "upper", "weight": 1.2, "description": "period-8 gradient imbalance"},
    "bicubic_up_high_freq_energy_ratio": {"tail": "lower", "weight": 1.2, "description": "reduced high-frequency energy"},
}

UPSAMPLE_CORE_FEATURES = [
    "bicubic_up_low_second_diff_energy",
    "bicubic_up_second_diff_phase8_range_ratio",
    "bicubic_up_second_diff_phase8_min_ratio",
    "bicubic_up_gradient_phase8_range_ratio",
    "bicubic_up_high_freq_energy_ratio",
]


def score_feature_group(
    group_name: str,
    feature_config: Dict[str, Dict],
    observed_features: Dict[str, float],
    null_features: List[Dict[str, float]],
    n_tests_per_feature: int,
    all_feature_results: Dict,
) -> float:
    total_score = 0.0

    for name, cfg in feature_config.items():
        obs = observed_features[name]
        null_vals = np.array([nf[name] for nf in null_features], dtype=np.float32)

        p, nfa = compute_nfa(
            observed=obs,
            null_values=null_vals,
            tail=cfg["tail"],
            n_tests=n_tests_per_feature,
        )

        score = nfa_to_score(nfa)
        weighted_score = cfg["weight"] * score
        total_score += weighted_score

        all_feature_results[name] = {
            "group": group_name,
            "description": cfg["description"],
            "observed": obs,
            "p_value": p,
            "NFA": nfa,
            "score": score,
            "weighted_score": weighted_score,
            "tail": cfg["tail"],
        }

    return total_score


def classify_features(
    observed_features: Dict[str, float],
    null_features: List[Dict[str, float]],
    n_tests_per_feature: int = 8,
    theta_jpeg: float = 3.0,
    theta_resample: float = 3.0,
    delta: float = 2.0,
) -> Dict:
    """
    Three-class decision:
        jpeg_compression
        upsample_x8
        original_or_uncertain
    """
    all_feature_results = {}

    jpeg_score = score_feature_group(
        "JPEG", JPEG_FEATURES, observed_features, null_features,
        n_tests_per_feature, all_feature_results,
    )
    generic_resample_score = score_feature_group(
        "Generic resampling", GENERIC_RESAMPLE_FEATURES, observed_features, null_features,
        n_tests_per_feature, all_feature_results,
    )
    upsample_score = score_feature_group(
        "Upsample x8", UPSAMPLE_X8_FEATURES, observed_features, null_features,
        n_tests_per_feature, all_feature_results,
    )

    combined_resample_score = 0.5*generic_resample_score + upsample_score

    upsample_hits = sum(
        1 for name in UPSAMPLE_X8_FEATURES
        if name in all_feature_results and all_feature_results[name]["NFA"] < 1.0
    )
    upsample_core_hits = sum(
        1 for name in UPSAMPLE_CORE_FEATURES
        if name in all_feature_results and all_feature_results[name]["NFA"] < 1.0
    )

    jpeg_detected = jpeg_score >= theta_jpeg
    upsample_detected = (
        (combined_resample_score >= theta_resample and upsample_hits >= 2)
        or (upsample_score >= theta_resample * 0.85 and upsample_core_hits >= 2)
    )

    if jpeg_detected and upsample_detected:
        if jpeg_score >= combined_resample_score + max(delta, 0.0):
            label = "jpeg_compression"
        else:
            label = "upsample_x8"
    elif jpeg_detected:
        label = "jpeg_compression"
    elif upsample_detected:
        label = "upsample_x8"
    else:
        label = "original_or_uncertain"

    return {
        "label": label,
        "jpeg_score": jpeg_score,
        "resample_score": combined_resample_score,
        "generic_resample_score": generic_resample_score,
        "upsample_score": upsample_score,
        "downsample_score": 0.0,
        "upsample_hits": upsample_hits,
        "upsample_core_hits": upsample_core_hits,
        "NFA_jpeg_final": 10 ** (-jpeg_score) if jpeg_score > 0 else 1.0,
        "NFA_resample_final": 10 ** (-combined_resample_score) if combined_resample_score > 0 else 1.0,
        "NFA_upsample_final": 10 ** (-upsample_score) if upsample_score > 0 else 1.0,
        "feature_results": all_feature_results,
    }


def print_report(result: Dict):
    print("\n================ Final Decision ================")
    print(f"Label: {result['label']}")
    print(f"JPEG score:       {result['jpeg_score']:.4f}")
    print(f"Resample score:   {result['resample_score']:.4f}")
    print(f"Generic R score:  {result['generic_resample_score']:.4f}")
    print(f"Upsample score:   {result['upsample_score']:.4f}")
    print(f"Upsample hits:    {result['upsample_hits']}")
    print(f"Upsample core hits: {result['upsample_core_hits']}")
    print(f"Final JPEG NFA:   {result['NFA_jpeg_final']:.4e}")
    print(f"Final R NFA:      {result['NFA_resample_final']:.4e}")
    print(f"Final Up NFA:     {result['NFA_upsample_final']:.4e}")

    print("\n================ Feature NFAs ================")
    for name, r in result["feature_results"].items():
        print(f"\n[{r['group']}] {name}")
        print(f"  Meaning:        {r['description']}")
        print(f"  Observed:       {r['observed']:.6f}")
        print(f"  Tail:           {r['tail']}")
        print(f"  p-value:        {r['p_value']:.6f}")
        print(f"  NFA:            {r['NFA']:.6f}")
        print(f"  Score:          {r['score']:.6f}")
        print(f"  Weighted score: {r['weighted_score']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="A contrario detector for JPEG compression vs bicubic upsample x8."
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--null_dir", type=str, default=None)
    parser.add_argument("--num_surrogates", type=int, default=64)
    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--max_null_images", type=int, default=30)
    parser.add_argument("--theta_jpeg", type=float, default=3.0)
    parser.add_argument("--theta_resample", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=2.0)

    args = parser.parse_args()

    img = load_grayscale_image(args.image, max_size=args.max_size)
    img = crop_to_multiple_of_8(img)

    print(f"[Info] Loaded image after top-left crop: {img.shape[1]}x{img.shape[0]}")
    print("[Info] Extracting observed features...")
    observed_features = extract_features(img, show_progress=True)

    if args.null_dir is not None:
        print("[Info] Building null distribution from clean image directory...")
        null_features = build_null_features_from_directory(
            args.null_dir,
            max_images=args.max_null_images,
            max_size=args.max_size,
        )
    else:
        print("[Info] Building null distribution from phase-randomized surrogates...")
        null_features = build_null_features_from_surrogates(
            img,
            num_surrogates=args.num_surrogates,
            seed=0,
        )

    print(f"[Info] Number of null samples: {len(null_features)}")

    result = classify_features(
        observed_features=observed_features,
        null_features=null_features,
        n_tests_per_feature=8,
        theta_jpeg=args.theta_jpeg,
        theta_resample=args.theta_resample,
        delta=args.delta,
    )

    print_report(result)


if __name__ == "__main__":
    main()
