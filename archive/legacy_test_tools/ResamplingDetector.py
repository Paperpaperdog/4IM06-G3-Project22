import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter
from scipy.stats import binom


@dataclass
class DetectionResult:
    axis: int
    image_size_along_axis: int
    distances: np.ndarray
    nfa: np.ndarray
    log10_nfa: np.ndarray
    k_values: np.ndarray
    detected: bool
    detected_distances: List[int]
    best_distance: Optional[int]
    best_nfa: float


def load_grayscale_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def rank_transform(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Rank transform used to suppress image content.

    For each pixel x, it counts how many pixels in its local neighborhood
    have lower intensity than x.

    This corresponds to:
        R(x) = sum_{x' in N(x)} 1{I(x') < I(x)}

    Parameters
    ----------
    image:
        2D grayscale image.
    window_size:
        Neighborhood size, default 7 as described in the paper.

    Returns
    -------
    transformed:
        Rank-transformed image normalized to [0, 1].
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    center_index = (window_size * window_size) // 2

    def rank_func(values: np.ndarray) -> float:
        center_value = values[center_index]
        return float(np.sum(values < center_value))

    transformed = generic_filter(
        image,
        function=rank_func,
        size=(window_size, window_size),
        mode="reflect",
    )

    max_rank = window_size * window_size - 1
    transformed = transformed / max_rank
    return transformed.astype(np.float32)


def tv_residual(image: np.ndarray, weight: float = 0.08) -> np.ndarray:
    """
    Extract residual using Total Variation denoising.

    This tries to remove smooth image content and keep residual components,
    which may contain noise and subtle resampling traces.

    Requires scikit-image. If unavailable, falls back to a simple
    high-pass residual.

    Parameters
    ----------
    image:
        2D grayscale image.
    weight:
        TV denoising strength.

    Returns
    -------
    residual:
        Residual image.
    """
    try:
        from skimage.restoration import denoise_tv_chambolle

        smooth = denoise_tv_chambolle(image, weight=weight, channel_axis=None)
        residual = image - smooth
    except Exception:
        warnings.warn(
            "scikit-image is not available. Falling back to simple high-pass residual."
        )
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(image, sigma=1.0)
        residual = image - smooth

    residual = residual - residual.mean()
    std = residual.std() + 1e-8
    residual = residual / std
    return residual.astype(np.float32)


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply preprocessing before spectral analysis.

    mode:
        - "rank": rank transform, suitable for uncompressed images
        - "tv": TV residual, useful for JPEG-compressed images
        - "none": no preprocessing
    """
    if mode == "rank":
        return rank_transform(image, window_size=7)
    if mode == "tv":
        return tv_residual(image)
    if mode == "none":
        return image.astype(np.float32)

    raise ValueError(f"Unknown preprocessing mode: {mode}")


def compute_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute centered 2D Fourier spectrum.

    The DFT output is complex-valued. fftshift moves the low-frequency
    component to the center, which is convenient for patch analysis.
    """
    spectrum = np.fft.fft2(image)
    spectrum = np.fft.fftshift(spectrum)
    return spectrum.astype(np.complex128)


def extract_non_overlapping_patches(
    spectrum: np.ndarray,
    patch_size: Tuple[int, int],
) -> np.ndarray:
    """
    Split a 2D complex spectrum into non-overlapping patches.

    Parameters
    ----------
    spectrum:
        2D complex Fourier spectrum.
    patch_size:
        (patch_height, patch_width)

    Returns
    -------
    patches:
        Array of shape (num_patches, patch_height * patch_width).
    """
    ph, pw = patch_size
    h, w = spectrum.shape

    h_crop = (h // ph) * ph
    w_crop = (w // pw) * pw
    cropped = spectrum[:h_crop, :w_crop]

    patches = cropped.reshape(h_crop // ph, ph, w_crop // pw, pw)
    patches = patches.transpose(0, 2, 1, 3)
    patches = patches.reshape(-1, ph * pw)

    return patches


def normalize_complex_patches(patches: np.ndarray) -> np.ndarray:
    """
    Mean-center and L2-normalize complex patches.

    This implements the normalization used by complex Pearson correlation:
        x_centered / ||x_centered||_2
    """
    centered = patches - patches.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return centered / norms


def complex_patch_correlation(
    normalized_anchor: np.ndarray,
    shifted_patches: np.ndarray,
) -> np.ndarray:
    """
    Compute complex Pearson correlation between anchor patches and shifted patches.

    corr(x, y) = | <x - mean(x), y - mean(y)> /
                   (||x - mean(x)||_2 ||y - mean(y)||_2) |

    Since anchor patches are already normalized, we normalize shifted patches
    and then compute the absolute complex inner product.
    """
    normalized_shifted = normalize_complex_patches(shifted_patches)

    # Complex inner product: sum x_i * conj(y_i)
    corr = np.abs(np.sum(normalized_anchor * np.conj(normalized_shifted), axis=1))
    return corr


def jpeg_suppressed_distances(n: int, radius: int = 3) -> set:
    """
    Return distances related to JPEG 8x8 block traces.

    JPEG tends to produce strong spectral correlations around:
        k * N / 8, k = 1, ..., 7

    The paper also suppresses nearby distances.
    """
    suppressed = set()

    for k in range(1, 8):
        center = int(round(k * n / 8.0))
        for d in range(center - radius, center + radius + 1):
            if 1 <= d < n:
                suppressed.add(d)

    return suppressed


def compute_rho_matrix_for_axis(
    spectrum: np.ndarray,
    axis: int,
    patch_size: Tuple[int, int],
    distances: np.ndarray,
) -> np.ndarray:
    """
    Compute rho(P_i, d) for all non-overlapping patches P_i
    and all candidate distances d along one axis.

    Parameters
    ----------
    spectrum:
        Complex Fourier spectrum.
    axis:
        0 for vertical direction, 1 for horizontal direction.
    patch_size:
        Size of each spectral patch.
    distances:
        Distances to evaluate.

    Returns
    -------
    rho:
        Matrix of shape (num_patches, num_distances).
        rho[i, j] is correlation of patch i at distance distances[j].
    """
    anchor_patches = extract_non_overlapping_patches(spectrum, patch_size)
    normalized_anchor = normalize_complex_patches(anchor_patches)

    rho = np.zeros((anchor_patches.shape[0], len(distances)), dtype=np.float64)

    for j, d in enumerate(distances):
        # Q_i is obtained by shifting the spectrum by d along the chosen axis.
        # np.roll implements periodicity, consistent with DFT periodicity.
        shifted_spectrum = np.roll(spectrum, shift=-int(d), axis=axis)
        shifted_patches = extract_non_overlapping_patches(shifted_spectrum, patch_size)
        rho[:, j] = complex_patch_correlation(normalized_anchor, shifted_patches)

    return rho


def binomial_tail(k: int, n: int, p: float) -> float:
    """
    Compute P[X >= k] for X ~ Binomial(n, p).

    scipy.stats.binom.sf(k - 1, n, p) gives P[X > k - 1] = P[X >= k].
    """
    if k <= 0:
        return 1.0
    return float(binom.sf(k - 1, n, p))


def detect_axis(
    spectrum: np.ndarray,
    axis: int = 0,
    patch_size: Tuple[int, int] = (8, 8),
    r: int = 3,
    epsilon: float = 1.0,
    suppress_jpeg: bool = False,
    jpeg_radius: int = 3,
    min_distance: int = 20,
) -> DetectionResult:
    """
    Detect anomalous spectral correlations along one axis using NFA.

    Main logic:
    1. For each candidate distance d, compute rho(P_i, d).
    2. Check whether rho(P_i, d) is a local maximum in [d-r, d+r].
    3. Count k(d), the number of patches where d is the local maximum.
    4. Under H0, k(d) ~ Binomial(#P, 1/(2r+1)).
    5. Compute NFA(d) = #C * P[X >= k(d)].
    6. If NFA(d) < epsilon, distance d is significant.

    Extra filtering:
    - If suppress_jpeg=True, skip JPEG-related distances near k*N/8.
    - Also skip candidate distances whose local comparison window touches
      JPEG-suppressed distances.
    - Skip very small distances and distances too close to N to avoid
      boundary-related false positives.

    Parameters
    ----------
    spectrum:
        2D Fourier spectrum.
    axis:
        0 vertical, 1 horizontal.
    patch_size:
        Spectral patch size.
    r:
        Local neighborhood radius for maximum comparison.
    epsilon:
        NFA threshold.
    suppress_jpeg:
        Whether to suppress JPEG-related distances.
    jpeg_radius:
        Suppression radius around k*N/8 distances.
    min_distance:
        Minimum allowed distance from 0 and from N.
        For example, min_distance=20 tests only distances:
            20 <= d <= N - 20

    Returns
    -------
    DetectionResult
    """
    n_axis = spectrum.shape[axis]

    if n_axis <= 2 * r + 2:
        raise ValueError("Image is too small along the selected axis.")

    if min_distance < r + 1:
        min_distance = r + 1

    if min_distance >= n_axis // 2:
        raise ValueError("min_distance is too large for this image size.")

    all_distances = np.arange(1, n_axis, dtype=np.int32)

    suppressed = (
        jpeg_suppressed_distances(n_axis, jpeg_radius)
        if suppress_jpeg
        else set()
    )

    # Build the actual tested distance set.
    # This avoids length mismatch when some distances are skipped.
    tested_distances = []

    for d in range(r + 1, n_axis - r):
        # Remove very small distances and distances close to the period boundary.
        # This helps suppress boundary/local-frequency false positives.
        if d < min_distance or d > n_axis - min_distance:
            continue

        # Skip direct JPEG-related distances.
        if suppress_jpeg and d in suppressed:
            continue

        # Skip distances whose local comparison window touches JPEG-related distances.
        # Otherwise boundary positions near the suppressed bands may become false maxima.
        window = list(range(d - r, d + r + 1))
        if suppress_jpeg and any(dp in suppressed for dp in window):
            continue

        tested_distances.append(d)

    tested_distances = np.asarray(tested_distances, dtype=np.int32)

    # Compute rho(P_i, d) for all possible distances.
    # We still need all distances because local windows [d-r, d+r] use neighbors.
    rho_all = compute_rho_matrix_for_axis(
        spectrum=spectrum,
        axis=axis,
        patch_size=patch_size,
        distances=all_distances,
    )

    num_patches = rho_all.shape[0]
    num_tests = len(tested_distances)

    if num_tests == 0:
        return DetectionResult(
            axis=axis,
            image_size_along_axis=n_axis,
            distances=tested_distances,
            nfa=np.array([], dtype=np.float64),
            log10_nfa=np.array([], dtype=np.float64),
            k_values=np.array([], dtype=np.int32),
            detected=False,
            detected_distances=[],
            best_distance=None,
            best_nfa=float("inf"),
        )

    p_local_max = 1.0 / (2 * r + 1)

    nfa_values = []
    k_values = []
    valid_distances = []

    for d in tested_distances:
        d = int(d)
        d_col = d - 1
        current_corr = rho_all[:, d_col]

        window = list(range(d - r, d + r + 1))

        neighbor_cols = []
        for dp in window:
            if 1 <= dp < n_axis:
                neighbor_cols.append(dp - 1)

        if len(neighbor_cols) != 2 * r + 1:
            continue

        neighbor_corr = rho_all[:, neighbor_cols]
        local_max = np.max(neighbor_corr, axis=1)

        # k(d): number of patches where distance d gives the local maximum
        k_d = int(np.sum(current_corr >= local_max))

        # Under H0:
        # k(d) ~ Binomial(#P, 1/(2r+1))
        tail = binomial_tail(k_d, num_patches, p_local_max)

        # NFA(d) = #C * P[X >= k(d)]
        nfa_d = num_tests * tail

        valid_distances.append(d)
        k_values.append(k_d)
        nfa_values.append(nfa_d)

    valid_distances = np.asarray(valid_distances, dtype=np.int32)
    nfa_values = np.asarray(nfa_values, dtype=np.float64)
    k_values = np.asarray(k_values, dtype=np.int32)

    with np.errstate(divide="ignore"):
        log10_nfa = np.log10(np.maximum(nfa_values, 1e-300))

    detected_mask = nfa_values < epsilon
    detected_distances = valid_distances[detected_mask].astype(int).tolist()

    if len(nfa_values) > 0:
        best_idx = int(np.argmin(nfa_values))
        best_distance = int(valid_distances[best_idx])
        best_nfa = float(nfa_values[best_idx])
    else:
        best_distance = None
        best_nfa = float("inf")

    return DetectionResult(
        axis=axis,
        image_size_along_axis=n_axis,
        distances=valid_distances,
        nfa=nfa_values,
        log10_nfa=log10_nfa,
        k_values=k_values,
        detected=bool(len(detected_distances) > 0),
        detected_distances=detected_distances,
        best_distance=best_distance,
        best_nfa=best_nfa,
    )


def cross_validate_proportional_resampling(
    result_axis0: DetectionResult,
    result_axis1: DetectionResult,
    beta: float = 0.01,
    epsilon: float = 1.0,
) -> Dict[str, object]:
    """
    Cross-validation for proportional resampling.

    If resizing preserves aspect ratio, then significant distances should
    have similar normalized ratios:

        d1 / N1 ~= d2 / N2

    This implements a practical version of Eq. (23):

        NFA_valid_1(d1) =
            max(NFA_1(d1),
                min_{|x/N2 - d1/N1| < beta} NFA_2(x))

    Parameters
    ----------
    result_axis0:
        Detection result along axis 0.
    result_axis1:
        Detection result along axis 1.
    beta:
        Allowed ratio difference.
    epsilon:
        NFA threshold.

    Returns
    -------
    dict containing validated distances.
    """
    n1 = result_axis0.image_size_along_axis
    n2 = result_axis1.image_size_along_axis

    validated = []

    for d1, nfa1 in zip(result_axis0.distances, result_axis0.nfa):
        ratio1 = d1 / n1

        ratios2 = result_axis1.distances / n2
        mask = np.abs(ratios2 - ratio1) < beta

        if not np.any(mask):
            continue

        min_nfa2 = float(np.min(result_axis1.nfa[mask]))
        best_d2 = int(result_axis1.distances[mask][np.argmin(result_axis1.nfa[mask])])

        valid_nfa = max(float(nfa1), min_nfa2)

        if valid_nfa < epsilon:
            validated.append(
                {
                    "d_axis0": int(d1),
                    "d_axis1": best_d2,
                    "ratio": float(ratio1),
                    "nfa_axis0": float(nfa1),
                    "nfa_axis1": float(min_nfa2),
                    "valid_nfa": float(valid_nfa),
                }
            )

    validated = sorted(validated, key=lambda item: item["valid_nfa"])

    return {
        "detected": len(validated) > 0,
        "validated_distances": validated,
    }

def make_result_path(
    prefix: str,
    axis: int,
    timestamp: str,
    output_dir: str = "results",
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    safe_prefix = Path(prefix).stem
    filename = f"{safe_prefix}_{timestamp}_axis{axis}.png"

    return str(Path(output_dir) / filename)

def save_nfa_plot(result: DetectionResult, output_path: str) -> None:
    """
    Save a line plot of log10 NFA over tested distances.
    The plot is saved to the given output path.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(result.distances, result.log10_nfa, linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Distance d")
    plt.ylabel("log10 NFA(d)")
    plt.title(f"NFA curve along axis {result.axis}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved NFA plot: {output_path}")


def print_result(result: DetectionResult, axis_name: str, top_k: int = 10) -> None:
    """
    Print detection summary.
    """
    print(f"\n=== Detection along {axis_name} ===")
    print(f"Axis size: {result.image_size_along_axis}")
    print(f"Detected: {result.detected}")
    print(f"Best distance: {result.best_distance}")
    print(f"Best NFA: {result.best_nfa:.6g}")

    if len(result.distances) == 0:
        print("No candidate distance tested.")
        return

    order = np.argsort(result.nfa)[:top_k]

    print(f"\nTop {top_k} suspicious distances:")
    print("distance\tk(d)\tNFA\t\tlog10(NFA)")
    for idx in order:
        print(
            f"{int(result.distances[idx])}\t\t"
            f"{int(result.k_values[idx])}\t"
            f"{result.nfa[idx]:.6g}\t"
            f"{result.log10_nfa[idx]:.3f}"
        )

def print_final_decision(
    results: Dict[int, DetectionResult],
    cv_result: Optional[Dict[str, object]] = None,
    epsilon: float = 1.0,
) -> None:
    """
    Print the final image-level decision.

    The paper's pipeline says:
        If NFA(d) < epsilon, then an abnormal correlation is detected at distance d.
        If at least one distance d is validated, the image is classified as resampled.

    If cross-validation is used, the final decision should preferably be based on
    the validated distances from cross-validation.
    """
    print("\n================ FINAL DECISION ================")

    # Case 1: cross-validation result is available
    if cv_result is not None:
        detected = bool(cv_result["detected"])

        if detected:
            best = cv_result["validated_distances"][0]

            print("Final result: RESAMPLED image")
            print("Reason: A validated abnormal spectral correlation was found.")
            print(f"Validated distance along axis 0: d0 = {best['d_axis0']}")
            print(f"Validated distance along axis 1: d1 = {best['d_axis1']}")
            print(f"Normalized distance ratio: {best['ratio']:.6f}")
            print(f"Validated NFA: {best['valid_nfa']:.6g}")
            print(f"Decision rule: NFA_valid(d) < epsilon = {epsilon}")

            print("\nPossible original size clues:")
            n0 = results[0].image_size_along_axis
            n1 = results[1].image_size_along_axis

            d0 = best["d_axis0"]
            d1 = best["d_axis1"]

            print(
                f"Axis 0: M0 may satisfy M0 ≡ ±{d0} (mod {n0}), "
                f"so possible clues include {d0} or {n0 - d0}."
            )
            print(
                f"Axis 1: M1 may satisfy M1 ≡ ±{d1} (mod {n1}), "
                f"so possible clues include {d1} or {n1 - d1}."
            )

        else:
            print("Final result: NOT confidently resampled")
            print("Reason: No distance passed bidirectional cross-validation.")
            print(f"Decision rule: no validated NFA_valid(d) < epsilon = {epsilon}")

        print("================================================")
        return

    # Case 2: no cross-validation; use per-axis detections
    detected_axes = []

    for axis, result in results.items():
        if result.detected:
            detected_axes.append(axis)

    if len(detected_axes) == 0:
        print("Final result: NOT resampled")
        print(f"Reason: No axis has any distance with NFA(d) < epsilon = {epsilon}.")
        print("================================================")
        return

    print("Final result: RESAMPLED image")
    print("Reason: At least one abnormal spectral correlation was detected.")
    print(f"Decision rule: NFA(d) < epsilon = {epsilon}")

    for axis in sorted(detected_axes):
        result = results[axis]
        print(f"\nAxis {axis}:")
        print(f"  Best abnormal distance: d = {result.best_distance}")
        print(f"  Best NFA: {result.best_nfa:.6g}")

        n = result.image_size_along_axis
        d = result.best_distance

        if d is not None:
            print(
                f"  Original size clue: M may satisfy M ≡ ±{d} (mod {n})."
            )
            print(
                f"  Simple candidates: M ≈ {d} or M ≈ {n - d}."
            )

    print("================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Core implementation of spectral-correlation image resampling detection."
    )
    parser.add_argument("image", type=str, help="Path to input image.")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="rank",
        choices=["rank", "tv", "none"],
        help="Preprocessing mode. Use rank for PNG/uncompressed, tv for JPEG.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=8,
        help="Square spectral patch size.",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="Local neighborhood radius for distance comparison.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="NFA threshold. A distance is significant if NFA < epsilon.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="both",
        choices=["0", "1", "both"],
        help="Axis to test: 0 vertical, 1 horizontal, both.",
    )
    parser.add_argument(
        "--suppress-jpeg",
        action="store_true",
        help="Suppress JPEG-related distances around k*N/8.",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Apply proportional-resampling cross validation between axes.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Tolerance for cross-validation ratio matching.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="If given, save log10 NFA plots with this prefix.",
    )
    parser.add_argument(
    "--jpeg-radius",
    type=int,
    default=3,
    help="Suppression radius around JPEG-related distances k*N/8.",
    )  
    parser.add_argument(
    "--min-distance",
    type=int,
    default=20,
    help="Ignore distances smaller than this value or closer than this value to N.",
)

    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image = load_grayscale_image(args.image)
    preprocessed = preprocess_image(image, args.preprocess)
    spectrum = compute_spectrum(preprocessed)

    patch_size = (args.patch_size, args.patch_size)

    print("Input image:", args.image)
    print("Image shape:", image.shape)
    print("Preprocessing:", args.preprocess)
    print("Patch size:", patch_size)
    print("r:", args.r)
    print("epsilon:", args.epsilon)
    print("JPEG suppression:", args.suppress_jpeg)

    results = {}

    if args.axis in ["0", "both"]:
        result0 = detect_axis(
            spectrum=spectrum,
            axis=0,
            patch_size=patch_size,
            r=args.r,
            epsilon=args.epsilon,
            suppress_jpeg=args.suppress_jpeg,
            jpeg_radius=args.jpeg_radius,
            min_distance=args.min_distance,
        )
        results[0] = result0
        print_result(result0, axis_name="axis 0 / vertical")

        if args.plot_prefix is not None:
            output_path = make_result_path(args.plot_prefix, axis=0, timestamp=run_timestamp)
            save_nfa_plot(result0, output_path)

    if args.axis in ["1", "both"]:
        result1 = detect_axis(
            spectrum=spectrum,
            axis=1,
            patch_size=patch_size,
            r=args.r,
            epsilon=args.epsilon,
            suppress_jpeg=args.suppress_jpeg,
            jpeg_radius=args.jpeg_radius,
            min_distance=args.min_distance,
        )
        results[1] = result1
        print_result(result1, axis_name="axis 1 / horizontal")

        if args.plot_prefix is not None:
            output_path = make_result_path(args.plot_prefix, axis=1, timestamp=run_timestamp)
            save_nfa_plot(result1, output_path)

    cv = None

    if args.cross_validate:
        if 0 not in results or 1 not in results:
            raise ValueError("--cross-validate requires --axis both.")

        cv = cross_validate_proportional_resampling(
            results[0],
            results[1],
            beta=args.beta,
            epsilon=args.epsilon,
        )

        print("\n=== Cross-validation for proportional resampling ===")
        print("Detected after cross-validation:", cv["detected"])

        if cv["validated_distances"]:
            print("d_axis0\td_axis1\tratio\t\tvalid_NFA")
            for item in cv["validated_distances"][:10]:
                print(
                    f"{item['d_axis0']}\t"
                    f"{item['d_axis1']}\t"
                    f"{item['ratio']:.5f}\t"
                    f"{item['valid_nfa']:.6g}"
                )

    print_final_decision(
        results=results,
        cv_result=cv,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()