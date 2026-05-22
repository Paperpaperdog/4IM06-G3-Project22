import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0


def rank_residual(img: np.ndarray, window: int = 7) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("Rank window must be odd and >= 3")
    r = window // 2
    h, w = img.shape
    if h <= 2 * r or w <= 2 * r:
        raise ValueError("Image is too small for selected rank window")

    core = img[r : h - r, r : w - r]
    count = np.zeros_like(core, dtype=np.float32)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            nbr = img[r + dy : h - r + dy, r + dx : w - r + dx]
            count += (core > nbr).astype(np.float32)

    return count / float(window * window - 1)


def highpass_residual(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return img - blur


def compute_spectrum(residual: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(residual)
    f = np.fft.fftshift(f)
    mag = np.log1p(np.abs(f))
    mag = mag - np.mean(mag)
    std = np.std(mag) + 1e-8
    return mag / std


def patchify(arr: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    h, w = arr.shape
    mh = h // patch_h
    mw = w // patch_w
    hh = mh * patch_h
    ww = mw * patch_w
    arr = arr[:hh, :ww]
    return arr.reshape(mh, patch_h, mw, patch_w).transpose(0, 2, 1, 3)


def corr_per_patch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b: [Mh, Mw, Ph, Pw]
    ax = a.reshape(a.shape[0], a.shape[1], -1)
    bx = b.reshape(b.shape[0], b.shape[1], -1)

    ax = ax - ax.mean(axis=-1, keepdims=True)
    bx = bx - bx.mean(axis=-1, keepdims=True)
    num = np.sum(ax * bx, axis=-1)
    den = np.sqrt(np.sum(ax * ax, axis=-1) * np.sum(bx * bx, axis=-1)) + 1e-10
    return num / den


def scan_distances(spec: np.ndarray, patch_h: int, patch_w: int, axis: str) -> np.ndarray:
    h, w = spec.shape
    max_d = h if axis == "vertical" else w
    all_corr = []
    for d in range(max_d):
        shifted = np.roll(spec, shift=d, axis=0 if axis == "vertical" else 1)
        pa = patchify(spec, patch_h, patch_w)
        pb = patchify(shifted, patch_h, patch_w)
        c = corr_per_patch(pa, pb).ravel()
        all_corr.append(c)
    return np.array(all_corr, dtype=np.float32)


def nfa_from_correlations(corr_mat: np.ndarray, ignore_radius: int) -> np.ndarray:
    mean_corr = corr_mat.mean(axis=1)
    d_max = mean_corr.shape[0]
    idx = np.arange(d_max)
    valid = (idx >= ignore_radius) & (idx <= d_max - ignore_radius - 1)
    base = mean_corr[valid] if np.any(valid) else mean_corr

    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    sigma = 1.4826 * mad + 1e-8
    z = (mean_corr - med) / sigma

    p = 1.0 - norm.cdf(z)
    n_tests = float(d_max)
    nfa = np.maximum(n_tests * p, 1e-300)
    return nfa


def detect(
    image_path: Path,
    outdir: Path,
    residual_mode: str,
    patch_h: int,
    patch_w: int,
    rank_window: int,
    nfa_threshold: float,
) -> None:
    img = load_grayscale(image_path)

    if residual_mode == "rank":
        residual = rank_residual(img, window=rank_window)
    elif residual_mode == "highpass":
        residual = highpass_residual(img)
    elif residual_mode == "none":
        residual = img.copy()
    else:
        raise ValueError(f"Unsupported residual mode: {residual_mode}")

    spec = compute_spectrum(residual)

    corr_v = scan_distances(spec, patch_h, patch_w, axis="vertical")
    corr_h = scan_distances(spec, patch_h, patch_w, axis="horizontal")

    ignore_radius = max(2, patch_h // 2)
    nfa_v = nfa_from_correlations(corr_v, ignore_radius=ignore_radius)
    nfa_h = nfa_from_correlations(corr_h, ignore_radius=ignore_radius)

    cand_v = nfa_v[ignore_radius : len(nfa_v) - ignore_radius]
    cand_h = nfa_h[ignore_radius : len(nfa_h) - ignore_radius]
    best_v = float(np.min(cand_v)) if cand_v.size else float(np.min(nfa_v))
    best_h = float(np.min(cand_h)) if cand_h.size else float(np.min(nfa_h))
    best = min(best_v, best_h)
    decision = best < nfa_threshold

    outdir.mkdir(parents=True, exist_ok=True)

    plot_path = outdir / f"{image_path.stem}_nfa.png"
    plt.figure(figsize=(10, 4))
    plt.semilogy(nfa_v, label="Vertical NFA")
    plt.semilogy(nfa_h, label="Horizontal NFA")
    plt.axhline(nfa_threshold, color="red", linestyle="--", linewidth=1.0, label=f"NFA={nfa_threshold:g}")
    plt.xlabel("Distance d (pixels in spectrum)")
    plt.ylabel("NFA (log scale)")
    plt.title("Resampling Detection NFA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Image: {image_path}")
    print(f"Residual mode: {residual_mode}")
    print(f"Patch size: {patch_h}x{patch_w}")
    print(f"Best vertical NFA: {best_v:.3e}")
    print(f"Best horizontal NFA: {best_h:.3e}")
    print(f"Decision threshold: {nfa_threshold:.3e}")
    print(f"Decision: {'RESAMPLED' if decision else 'NOT_RESAMPLED'}")
    print(f"NFA plot saved to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised resampling detector with NFA plot output.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Directory to save outputs")
    parser.add_argument("--residual", choices=["rank", "highpass", "none"], default="rank")
    parser.add_argument("--rank-window", type=int, default=7, help="Odd window size for rank residual")
    parser.add_argument("--patch-h", type=int, default=16, help="Patch height in spectrum")
    parser.add_argument("--patch-w", type=int, default=16, help="Patch width in spectrum")
    parser.add_argument("--nfa-threshold", type=float, default=1e-3, help="Decision threshold for NFA")
    args = parser.parse_args()

    detect(
        image_path=args.image,
        outdir=args.outdir,
        residual_mode=args.residual,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        rank_window=args.rank_window,
        nfa_threshold=args.nfa_threshold,
    )


if __name__ == "__main__":
    main()
