"""Generate Fig. 6 — radial power-profile overlay (1-D) for Section 4.4.

Demonstrates the Fourier ambiguity: the radial log-magnitude spectra of
downsample x8 and x16 nearly coincide, while upsample x4 / x8 separate cleanly.

Preprocessing is reused verbatim from the n6 pipeline
(`experiments.unified_protocol` + `src.processing`), so the spectra match the
ones the CNN/Mask models were trained on. Because resampling traces are a
deterministic function of the scaling factor and largely content-independent,
many random crops from a single high-resolution RAISE TIFF are sufficient to
recover the per-class mean radial profile for a *qualitative* figure.

Run:
    spectral-mask-resampling/.venv/bin/python scripts/analysis/make_radial_profile_fig.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
CNN_ROOT = REPO / "CNN" / "spectral-history-cnn"
for p in (str(REPO), str(CNN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.unified_protocol import make_aligned_observed_patch  # noqa: E402
from src.processing.residuals import tv_residual  # noqa: E402
from src.processing.spectrum import compute_log_rfft_spectrum  # noqa: E402
from src.processing.transforms import rgb_to_y_float  # noqa: E402

# ---- n6 size-128 protocol parameters (from configs/size_sweep/n6_poscnn_size128.yaml)
OBSERVED_SIZE = 128
JPEG_QUALITY = 80
TV_WEIGHT = 0.08
TV_EPS = 1e-4
TV_MAX_ITER = 30
DC_SIGMA_BINS = 3.0
GLOBAL_SEED = 123
SPLIT = "test"

N_PER_CLASS = 200  # crops averaged per class
CLASSES = ["original", "JPEG_Q80", "downsample_x8", "downsample_x16", "upsample_x4", "upsample_x8"]
TIFF = REPO / "spectral-mask-resampling" / "data" / "raw" / "raise_tiff" / "r24dcc781t.TIF"

OUT_DIR = REPO / "results" / "comparison" / "radial_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def spectrum_for(class_name: str, class_index: int, sample_index: int, src_img: Image.Image) -> np.ndarray | None:
    patch = make_aligned_observed_patch(
        ["__single__"],
        lambda _e: src_img,
        class_name,
        OBSERVED_SIZE,
        GLOBAL_SEED,
        SPLIT,
        class_index,
        sample_index,
        JPEG_QUALITY,
    )
    if patch is None:
        return None
    y = rgb_to_y_float(patch)
    residual = tv_residual(y, weight=TV_WEIGHT, eps=TV_EPS, max_num_iter=TV_MAX_ITER)
    spec = compute_log_rfft_spectrum(residual, dc_sigma_bins=DC_SIGMA_BINS)  # (1,H,Wr)
    return np.squeeze(spec, axis=0).astype(np.float64)


def radial_profile(spec2d: np.ndarray, n_bins: int = 64):
    """Azimuthal average of a (H, Wr) fftshift(v)/rfft(u) spectrum vs normalized radius."""
    h, wr = spec2d.shape
    u = np.fft.rfftfreq(h)                       # 0 .. 0.5  (width == height here)
    v = np.fft.fftshift(np.fft.fftfreq(h))       # -0.5 .. 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")
    r = np.sqrt(uu ** 2 + vv ** 2)
    r_max = 0.5
    bins = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    vals = spec2d.ravel()
    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            prof[b] = vals[m].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof


def main() -> None:
    print(f"Loading source TIFF: {TIFF.name}")
    src_img = Image.open(TIFF).convert("RGB")

    mean_spec = {}
    for ci, cname in enumerate(CLASSES):
        acc = None
        made = 0
        for si in range(N_PER_CLASS * 2):  # extra attempts in case of skips
            if made >= N_PER_CLASS:
                break
            s = spectrum_for(cname, ci, si, src_img)
            if s is None:
                continue
            acc = s if acc is None else acc + s
            made += 1
        mean_spec[cname] = acc / made
        print(f"  {cname:16s}: averaged {made} crops, spectrum {mean_spec[cname].shape}")

    # radial profiles (mean log-magnitude vs radius)
    profiles = {}
    centers = None
    for cname in CLASSES:
        centers, prof = radial_profile(mean_spec[cname])
        profiles[cname] = prof

    # cumulative radial energy (CDF over radius) -> band-limited vs broadband
    def cdf_and_r90(prof: np.ndarray):
        p = np.nan_to_num(prof, nan=0.0)
        p = np.clip(p, 0.0, None)
        total = p.sum()
        cdf = np.cumsum(p) / total if total > 0 else np.zeros_like(p)
        r90 = float(np.interp(0.9, cdf, centers))  # effective bandwidth (90% energy)
        return cdf, r90

    cdfs, r90 = {}, {}
    for cname in CLASSES:
        cdfs[cname], r90[cname] = cdf_and_r90(profiles[cname])

    def cosine(a, b):
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_down = cosine(profiles["downsample_x8"], profiles["downsample_x16"])
    cos_up = cosine(profiles["upsample_x4"], profiles["upsample_x8"])

    # save numeric outputs
    np.savez(OUT_DIR / "radial_profiles.npz", centers=centers,
             **{f"prof_{k}": profiles[k] for k in CLASSES},
             **{f"cdf_{k}": cdfs[k] for k in CLASSES})
    import csv

    with open(OUT_DIR / "radial_profiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radial_freq"] + [f"prof_{c}" for c in CLASSES] + [f"cdf_{c}" for c in CLASSES])
        for i, c in enumerate(centers):
            w.writerow([f"{c:.5f}"]
                       + [f"{profiles[k][i]:.6f}" for k in CLASSES]
                       + [f"{cdfs[k][i]:.6f}" for k in CLASSES])

    style = {
        "downsample_x8":  dict(color="#d1495b", lw=2.3, ls="-",  label="Down×8"),
        "downsample_x16": dict(color="#edae49", lw=2.3, ls="-",  label="Down×16"),
        "upsample_x4":    dict(color="#2e86ab", lw=2.1, ls="--", label="Up×4"),
        "upsample_x8":    dict(color="#3fa34d", lw=2.1, ls="--", label="Up×8"),
    }

    # ---- Figure: two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel (a): radial log-magnitude profile
    for cname, st in style.items():
        ax1.plot(centers, profiles[cname], **st)
    ax1.set_title("(a) Radial log-magnitude profile", fontsize=11)
    ax1.set_xlabel("Normalized radial frequency (cycles/pixel)")
    ax1.set_ylabel("Mean log-magnitude")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    # Panel (b): cumulative radial energy (the honest discriminator)
    for cname, st in style.items():
        ax2.plot(centers, cdfs[cname], **st)
    # mark effective bandwidth (r90) for the up classes vs the down pair
    ax2.axhline(0.9, color="gray", lw=0.8, ls=":")
    for cname in ["upsample_x8", "upsample_x4", "downsample_x8", "downsample_x16"]:
        ax2.plot([r90[cname]], [0.9], marker="o", ms=5, color=style[cname]["color"])
    ax2.set_title("(b) Cumulative radial energy (band-limited vs broadband)", fontsize=10.5)
    ax2.set_xlabel("Normalized radial frequency (cycles/pixel)")
    ax2.set_ylabel("Fraction of energy below radius")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(alpha=0.25)

    # headline quantitative result: profile cosine similarity
    ax1.text(0.97, 0.50,
             f"profile cosine:\nDown×8–Down×16 = {cos_down:.3f}\nUp×4–Up×8 = {cos_up:.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    fig.suptitle(
        "Radial spectra (n6, size 128): Up classes are band-limited (separable), "
        "Down×8 ≈ Down×16 are broadband (ambiguous)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT_DIR / "radial_profile_overlay.png"
    fig.savefig(out_png, dpi=200)

    print(f"\nSaved figure: {out_png}")
    print(f"Saved data:   {OUT_DIR / 'radial_profiles.csv'}")
    print("\n--- effective bandwidth r90 (radius holding 90% of radial energy) ---")
    for cname in CLASSES:
        print(f"  {cname:16s}: r90 = {r90[cname]:.4f}")
    print("\n--- profile cosine similarity (shape) ---")
    print(f"  Down×8 vs Down×16 = {cos_down:.4f}   (high => ambiguous pair)")
    print(f"  Up×4   vs Up×8    = {cos_up:.4f}")
    print(f"  r90 ratio Down16/Down8 = {r90['downsample_x16']/r90['downsample_x8']:.2f}  "
          f"(≈1 => indistinguishable bandwidth)")
    print(f"  r90 ratio Up4/Up8      = {r90['upsample_x4']/r90['upsample_x8']:.2f}  "
          f"(>>1 => clearly different bandwidth)")


if __name__ == "__main__":
    main()
