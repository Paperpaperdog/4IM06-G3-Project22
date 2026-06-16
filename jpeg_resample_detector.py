import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import convolve, median_filter


# ============================================================
# Basic image utilities
# ============================================================

def load_grayscale_image(path: str, max_size: int = 512) -> np.ndarray:
    """
    Load an image as grayscale float32 array in [0, 255].
    The image is optionally resized so that the maximum side length is max_size.
    """
    img = Image.open(path).convert("L")

    w, h = img.size
    scale = max(w, h) / max_size

    if scale > 1:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32)
    return arr


def crop_to_multiple_of_8(img: np.ndarray) -> np.ndarray:
    """
    Crop image so that height and width are multiples of 8.
    """
    h, w = img.shape
    h8 = h - h % 8
    w8 = w - w % 8
    return img[:h8, :w8]


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to zero mean and unit variance.
    """
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)


# ============================================================
# Residual image
# ============================================================

def prediction_residual(img: np.ndarray) -> np.ndarray:
    """
    Compute a simple prediction residual.

    r(x, y) = I(x, y) - mean of 4-neighborhood

    JPEG artifacts and resampling artifacts are often clearer in residuals
    than in raw images.
    """
    kernel = np.array([
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0]
    ], dtype=np.float32)

    pred = convolve(img, kernel, mode="reflect")
    residual = img - pred
    return residual


# ============================================================
# FFT spectrum features
# ============================================================

def log_fft_spectrum(img: np.ndarray) -> np.ndarray:
    """
    Compute centered log magnitude FFT spectrum.
    """
    F = np.fft.fftshift(np.fft.fft2(img))
    S = np.log1p(np.abs(F))
    return S.astype(np.float32)


def locally_normalize_spectrum(S: np.ndarray, size: int = 31) -> np.ndarray:
    """
    Remove local background from spectrum by median filtering.

    This makes peaks more comparable across frequencies.
    """
    bg = median_filter(S, size=size, mode="reflect")
    S_norm = S - bg
    return S_norm


def get_8_periodic_peak_positions(shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Return common 8-periodic spectral positions.

    Important:
    These positions are NOT used to distinguish JPEG vs 8x8 resampling.
    They are only used to measure peak shape features such as sharpness and width.
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    positions = []

    # Frequencies corresponding to period 8.
    dx = max(1, w // 8)
    dy = max(1, h // 8)

    # Horizontal and vertical symmetric positions.
    candidates = [
        (cy, cx + dx),
        (cy, cx - dx),
        (cy + dy, cx),
        (cy - dy, cx),

        # Harmonics can be useful.
        (cy, cx + 2 * dx),
        (cy, cx - 2 * dx),
        (cy + 2 * dy, cx),
        (cy - 2 * dy, cx),
    ]

    for y, x in candidates:
        if 0 <= y < h and 0 <= x < w:
            positions.append((y, x))

    return positions


def peak_shape_features(S_norm: np.ndarray, radius: int = 4) -> Dict[str, float]:
    """
    Compute peak sharpness and peak width around 8-periodic spectral peaks.

    JPEG tends to produce sharper, more concentrated block-grid peaks.
    8x8 resampling/interpolation tends to produce wider or more spread peaks.

    sharpness = max / local_mean
    width     = local_sum / max
    """
    h, w = S_norm.shape
    positions = get_8_periodic_peak_positions(S_norm.shape)

    sharpness_values = []
    width_values = []

    for y, x in positions:
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)

        patch = S_norm[y0:y1, x0:x1]
        patch_shifted = patch - patch.min() + 1e-8

        peak = float(patch_shifted.max())
        mean_val = float(patch_shifted.mean())
        sum_val = float(patch_shifted.sum())

        sharpness = peak / (mean_val + 1e-8)
        width = sum_val / (peak + 1e-8)

        sharpness_values.append(sharpness)
        width_values.append(width)

    return {
        "peak_sharpness": float(np.mean(sharpness_values)),
        "peak_width": float(np.mean(width_values)),
    }


# ============================================================
# JPEG-specific spatial block features
# ============================================================

def block_boundary_ratio(img: np.ndarray) -> float:
    """
    Measure 8-pixel block boundary discontinuity.

    JPEG compression usually increases differences across 8x8 block boundaries.
    Resampling may create 8-periodicity, but not necessarily sharp block boundaries.
    """
    img = img.astype(np.float32)
    h, w = img.shape

    # Vertical boundaries: difference across columns x-1 and x where x % 8 == 0.
    if w > 9:
        vertical_diff = np.abs(img[:, 1:] - img[:, :-1])
        cols = np.arange(1, w)
        boundary_cols = cols % 8 == 0
        non_boundary_cols = ~boundary_cols

        b_v = vertical_diff[:, boundary_cols].mean() if boundary_cols.any() else 0.0
        nb_v = vertical_diff[:, non_boundary_cols].mean() if non_boundary_cols.any() else 1e-8
    else:
        b_v, nb_v = 0.0, 1e-8

    # Horizontal boundaries: difference across rows y-1 and y where y % 8 == 0.
    if h > 9:
        horizontal_diff = np.abs(img[1:, :] - img[:-1, :])
        rows = np.arange(1, h)
        boundary_rows = rows % 8 == 0
        non_boundary_rows = ~boundary_rows

        b_h = horizontal_diff[boundary_rows, :].mean() if boundary_rows.any() else 0.0
        nb_h = horizontal_diff[non_boundary_rows, :].mean() if non_boundary_rows.any() else 1e-8
    else:
        b_h, nb_h = 0.0, 1e-8

    boundary = 0.5 * (b_v + b_h)
    non_boundary = 0.5 * (nb_v + nb_h)

    return float(boundary / (non_boundary + 1e-8))


# ============================================================
# DCT features for JPEG quantization
# ============================================================

def block_dct_8x8(img: np.ndarray) -> np.ndarray:
    """
    Compute 8x8 block DCT coefficients.

    Returns:
        coeffs: shape [num_blocks_y, num_blocks_x, 8, 8]
    """
    img = crop_to_multiple_of_8(img)
    h, w = img.shape

    blocks = img.reshape(h // 8, 8, w // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3)

    coeffs = dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    return coeffs.astype(np.float32)


def dct_high_frequency_zero_ratio(coeffs: np.ndarray, zero_threshold: float = 1.0) -> float:
    """
    JPEG quantization often creates many near-zero high-frequency DCT coefficients.

    Since the image has usually been decoded into pixels, coefficients may not be
    exactly zero, so we use a small threshold.
    """
    hf_mask = np.zeros((8, 8), dtype=bool)

    for u in range(8):
        for v in range(8):
            if u + v >= 8:
                hf_mask[u, v] = True

    hf_coeffs = coeffs[:, :, hf_mask]
    zero_ratio = np.mean(np.abs(hf_coeffs) < zero_threshold)

    return float(zero_ratio)


def dct_histogram_comb_score(coeffs: np.ndarray, bins: int = 80) -> float:
    """
    Measure comb-like irregularity of DCT coefficient histograms.

    JPEG quantization tends to create histogram gaps and periodic structures.
    We approximate this by the normalized second difference of histograms.

    Larger value means stronger quantization-like histogram irregularity.
    """
    selected_positions = [
        (0, 1), (1, 0),
        (1, 1), (0, 2), (2, 0),
        (2, 1), (1, 2),
        (3, 0), (0, 3),
    ]

    scores = []

    for u, v in selected_positions:
        c = coeffs[:, :, u, v].ravel()

        # Robust histogram range.
        lo, hi = np.percentile(c, [1, 99])
        if hi <= lo + 1e-8:
            continue

        hist, _ = np.histogram(c, bins=bins, range=(lo, hi))
        hist = hist.astype(np.float32)

        if hist.mean() <= 1e-8:
            continue

        second_diff = np.diff(hist, n=2)
        roughness = np.mean(np.abs(second_diff)) / (hist.mean() + 1e-8)
        scores.append(float(roughness))

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# Resampling-specific features
# ============================================================

def residual_autocorrelation_lag8(residual: np.ndarray) -> float:
    """
    Measure normalized residual autocorrelation at lag 8.

    8x8 resampling/interpolation may introduce periodic correlations.
    """
    r = normalize_image(residual)

    vals = []

    if r.shape[1] > 8:
        a = r[:, :-8]
        b = r[:, 8:]
        vals.append(float(np.mean(a * b)))

    if r.shape[0] > 8:
        a = r[:-8, :]
        b = r[8:, :]
        vals.append(float(np.mean(a * b)))

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))


def second_difference_energy(img: np.ndarray) -> float:
    """
    Measure local second-order variation.

    Interpolation-based resampling tends to smooth local second differences.
    JPEG tends to introduce boundary discontinuities instead.
    """
    img = img.astype(np.float32)

    dx2 = img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2]
    dy2 = img[2:, :] - 2 * img[1:-1, :] + img[:-2, :]

    energy_x = np.mean(np.abs(dx2))
    energy_y = np.mean(np.abs(dy2))

    return float(0.5 * (energy_x + energy_y))


def boundary_second_difference_ratio(img: np.ndarray) -> float:
    """
    Compare second-difference energy near 8-pixel boundaries and non-boundaries.

    JPEG usually has stronger second-difference energy around block boundaries.
    8x8 resampling tends to be smoother and less boundary-discontinuous.
    """
    img = img.astype(np.float32)
    h, w = img.shape

    ratios = []

    if w > 10:
        dx2 = np.abs(img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2])
        cols = np.arange(1, w - 1)
        boundary_cols = cols % 8 == 0
        non_boundary_cols = ~boundary_cols

        if boundary_cols.any() and non_boundary_cols.any():
            b = dx2[:, boundary_cols].mean()
            nb = dx2[:, non_boundary_cols].mean()
            ratios.append(float(b / (nb + 1e-8)))

    if h > 10:
        dy2 = np.abs(img[2:, :] - 2 * img[1:-1, :] + img[:-2, :])
        rows = np.arange(1, h - 1)
        boundary_rows = rows % 8 == 0
        non_boundary_rows = ~boundary_rows

        if boundary_rows.any() and non_boundary_rows.any():
            b = dy2[boundary_rows, :].mean()
            nb = dy2[non_boundary_rows, :].mean()
            ratios.append(float(b / (nb + 1e-8)))

    if len(ratios) == 0:
        return 1.0

    return float(np.mean(ratios))


# ============================================================
# Feature extraction
# ============================================================

def extract_features(img: np.ndarray) -> Dict[str, float]:
    """
    Extract all features used by the detector.
    """
    img = crop_to_multiple_of_8(img)

    residual = prediction_residual(img)
    S = log_fft_spectrum(residual)
    S_norm = locally_normalize_spectrum(S)

    coeffs = block_dct_8x8(img)
    peak_feats = peak_shape_features(S_norm)

    features = {
        # JPEG-specific evidence
        "block_boundary_ratio": block_boundary_ratio(img),
        "dct_hf_zero_ratio": dct_high_frequency_zero_ratio(coeffs),
        "dct_comb_score": dct_histogram_comb_score(coeffs),
        "peak_sharpness": peak_feats["peak_sharpness"],

        # Resampling-specific evidence
        "residual_autocorr_lag8": residual_autocorrelation_lag8(residual),
        "peak_width": peak_feats["peak_width"],
        "second_diff_energy": second_difference_energy(img),
        "boundary_second_diff_ratio": boundary_second_difference_ratio(img),
    }

    return features


# ============================================================
# Null hypothesis generation
# ============================================================

def phase_randomized_surrogate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a surrogate image under a null hypothesis.

    This keeps the global Fourier magnitude approximately similar,
    but randomizes phase, destroying structured JPEG or resampling artifacts.

    This is useful when no clean reference dataset is available.
    """
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std() + 1e-8

    F = np.fft.fft2(img)
    magnitude = np.abs(F)

    random_phase = rng.uniform(-np.pi, np.pi, size=img.shape)
    F_random = magnitude * np.exp(1j * random_phase)

    surrogate = np.real(np.fft.ifft2(F_random))

    # Match mean and std of original image.
    surrogate = (surrogate - surrogate.mean()) / (surrogate.std() + 1e-8)
    surrogate = surrogate * std + mean

    surrogate = np.clip(surrogate, 0, 255).astype(np.float32)
    return surrogate


def build_null_features_from_surrogates(
    img: np.ndarray,
    num_surrogates: int = 64,
    seed: int = 0
) -> List[Dict[str, float]]:
    """
    Build empirical null distribution by phase-randomized surrogates.
    """
    rng = np.random.default_rng(seed)
    null_features = []

    for _ in range(num_surrogates):
        surr = phase_randomized_surrogate(img, rng)
        feats = extract_features(surr)
        null_features.append(feats)

    return null_features


def build_null_features_from_directory(
    null_dir: str,
    max_images: int = 100,
    max_size: int = 512
) -> List[Dict[str, float]]:
    """
    Build empirical null distribution from a directory of original / clean images.

    Recommended if you have clean non-JPEG, non-resampled images.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = []

    for p in Path(null_dir).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))

    paths = paths[:max_images]

    null_features = []

    for p in paths:
        try:
            img = load_grayscale_image(p, max_size=max_size)
            feats = extract_features(img)
            null_features.append(feats)
        except Exception as e:
            print(f"[Warning] Failed to process null image {p}: {e}")

    if len(null_features) == 0:
        raise ValueError("No valid null images found in null_dir.")

    return null_features


# ============================================================
# NFA computation
# ============================================================

def empirical_p_value(
    observed: float,
    null_values: np.ndarray,
    tail: str
) -> float:
    """
    Empirical p-value with +1 smoothing.

    tail = "upper":
        p = P(F >= observed)

    tail = "lower":
        p = P(F <= observed)
    """
    n = len(null_values)

    if tail == "upper":
        count = np.sum(null_values >= observed)
    elif tail == "lower":
        count = np.sum(null_values <= observed)
    else:
        raise ValueError("tail must be 'upper' or 'lower'")

    return float((count + 1) / (n + 1))


def compute_nfa(
    observed: float,
    null_values: np.ndarray,
    tail: str,
    n_tests: int
) -> Tuple[float, float]:
    """
    Compute p-value and NFA.

    NFA = N_tests * p_value
    """
    p = empirical_p_value(observed, null_values, tail)
    nfa = n_tests * p
    return p, float(nfa)


def nfa_to_score(nfa: float) -> float:
    """
    Convert NFA to evidence score.

    Only NFA < 1 contributes positive evidence.
    """
    nfa = max(float(nfa), 1e-300)

    if nfa >= 1.0:
        return 0.0

    return float(-np.log10(nfa))


# ============================================================
# Classification
# ============================================================

JPEG_FEATURES = {
    "block_boundary_ratio": {
        "tail": "upper",
        "weight": 1.5,
        "description": "8-pixel block boundary discontinuity"
    },
    "dct_hf_zero_ratio": {
        "tail": "upper",
        "weight": 1.5,
        "description": "high-frequency DCT zero ratio"
    },
    "dct_comb_score": {
        "tail": "upper",
        "weight": 1.2,
        "description": "DCT histogram comb-like quantization"
    },
    "peak_sharpness": {
        "tail": "upper",
        "weight": 1.0,
        "description": "sharpness of 8-periodic spectral peaks"
    },
}

RESAMPLE_FEATURES = {
    "residual_autocorr_lag8": {
        "tail": "upper",
        "weight": 1.5,
        "description": "residual autocorrelation at lag 8"
    },
    "peak_width": {
        "tail": "upper",
        "weight": 1.0,
        "description": "width/spreading of 8-periodic spectral peaks"
    },
    "second_diff_energy": {
        "tail": "lower",
        "weight": 1.0,
        "description": "low second-difference energy caused by interpolation smoothness"
    },
    "boundary_second_diff_ratio": {
        "tail": "lower",
        "weight": 1.2,
        "description": "weak block-boundary second-difference discontinuity"
    },
}


def classify_features(
    observed_features: Dict[str, float],
    null_features: List[Dict[str, float]],
    n_tests_per_feature: int = 8,
    theta_jpeg: float = 3.0,
    theta_resample: float = 3.0,
    delta: float = 2.0
) -> Dict:
    """
    Compute feature-wise NFAs and final decision.

    Final scores:

        Score_J = sum_i w_i * [-log10(NFA_J_i)]
        Score_R = sum_i v_i * [-log10(NFA_R_i)]

    Decision:
        JPEG if JPEG score is high and clearly stronger.
        8x8 resampling if resampling score is high and clearly stronger.
        JPEG + resampling if both are high and close.
        original / uncertain otherwise.
    """
    all_feature_results = {}

    jpeg_score = 0.0
    resample_score = 0.0

    # JPEG NFAs
    for name, cfg in JPEG_FEATURES.items():
        obs = observed_features[name]
        null_vals = np.array([nf[name] for nf in null_features], dtype=np.float32)

        p, nfa = compute_nfa(
            observed=obs,
            null_values=null_vals,
            tail=cfg["tail"],
            n_tests=n_tests_per_feature
        )

        score = nfa_to_score(nfa)
        weighted_score = cfg["weight"] * score
        jpeg_score += weighted_score

        all_feature_results[name] = {
            "group": "JPEG",
            "description": cfg["description"],
            "observed": obs,
            "p_value": p,
            "NFA": nfa,
            "score": score,
            "weighted_score": weighted_score,
            "tail": cfg["tail"]
        }

    # Resampling NFAs
    for name, cfg in RESAMPLE_FEATURES.items():
        obs = observed_features[name]
        null_vals = np.array([nf[name] for nf in null_features], dtype=np.float32)

        p, nfa = compute_nfa(
            observed=obs,
            null_values=null_vals,
            tail=cfg["tail"],
            n_tests=n_tests_per_feature
        )

        score = nfa_to_score(nfa)
        weighted_score = cfg["weight"] * score
        resample_score += weighted_score

        all_feature_results[name] = {
            "group": "Resampling",
            "description": cfg["description"],
            "observed": obs,
            "p_value": p,
            "NFA": nfa,
            "score": score,
            "weighted_score": weighted_score,
            "tail": cfg["tail"]
        }

    # Convert scores back to final pseudo-NFA.
    nfa_jpeg_final = 10 ** (-jpeg_score) if jpeg_score > 0 else 1.0
    nfa_resample_final = 10 ** (-resample_score) if resample_score > 0 else 1.0

    # Final decision.
    if jpeg_score >= theta_jpeg and resample_score >= theta_resample:
        if abs(jpeg_score - resample_score) <= delta:
            label = "jpeg_and_8x8_resampling"
        elif jpeg_score > resample_score:
            label = "jpeg_dominant_possible_jpeg_plus_resampling"
        else:
            label = "resampling_dominant_possible_jpeg_plus_resampling"

    elif jpeg_score >= theta_jpeg and jpeg_score - resample_score > delta:
        label = "jpeg_compression"

    elif resample_score >= theta_resample and resample_score - jpeg_score > delta:
        label = "8x8_resampling"

    else:
        label = "original_or_uncertain"

    return {
        "label": label,
        "jpeg_score": jpeg_score,
        "resample_score": resample_score,
        "NFA_jpeg_final": nfa_jpeg_final,
        "NFA_resample_final": nfa_resample_final,
        "feature_results": all_feature_results
    }


# ============================================================
# Printing utilities
# ============================================================

def print_report(result: Dict):
    print("\n================ Final Decision ================")
    print(f"Label: {result['label']}")
    print(f"JPEG score:      {result['jpeg_score']:.4f}")
    print(f"Resample score:  {result['resample_score']:.4f}")
    print(f"Final JPEG NFA:  {result['NFA_jpeg_final']:.4e}")
    print(f"Final R NFA:     {result['NFA_resample_final']:.4e}")

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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="A contrario detector for JPEG compression vs 8x8 resampling."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image."
    )

    parser.add_argument(
        "--null_dir",
        type=str,
        default=None,
        help="Optional directory of clean/original images for empirical null distribution."
    )

    parser.add_argument(
        "--num_surrogates",
        type=int,
        default=64,
        help="Number of phase-randomized surrogate images if null_dir is not provided."
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=512,
        help="Maximum image side length used for processing."
    )

    parser.add_argument(
        "--theta_jpeg",
        type=float,
        default=3.0,
        help="JPEG score threshold."
    )

    parser.add_argument(
        "--theta_resample",
        type=float,
        default=3.0,
        help="Resampling score threshold."
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Minimum score difference for single-class decision."
    )

    args = parser.parse_args()

    img = load_grayscale_image(args.image, max_size=args.max_size)
    img = crop_to_multiple_of_8(img)

    print("[Info] Extracting observed features...")
    observed_features = extract_features(img)

    if args.null_dir is not None:
        print("[Info] Building null distribution from clean image directory...")
        null_features = build_null_features_from_directory(
            args.null_dir,
            max_images=100,
            max_size=args.max_size
        )
    else:
        print("[Info] Building null distribution from phase-randomized surrogates...")
        null_features = build_null_features_from_surrogates(
            img,
            num_surrogates=args.num_surrogates,
            seed=0
        )

    print(f"[Info] Number of null samples: {len(null_features)}")

    result = classify_features(
        observed_features=observed_features,
        null_features=null_features,
        n_tests_per_feature=8,
        theta_jpeg=args.theta_jpeg,
        theta_resample=args.theta_resample,
        delta=args.delta
    )

    print_report(result)


if __name__ == "__main__":
    main()