import numpy as np
import torch


def build_dc_weight(height: int, width: int, dc_sigma_bins: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Radial Gaussian DC-suppression weight on the native rFFT grid.

    Identical to the CNN route (CNN/spectral-history-cnn/src/processing/spectrum.py)
    so the two methods consume the same spectral representation.
    """
    u = torch.fft.rfftfreq(width, d=1.0, device=device).to(dtype)
    v = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype)
    v = torch.fft.fftshift(v)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    r_bins = torch.sqrt((u_grid * width) ** 2 + (v_grid * height) ** 2 + 1e-12)
    dc_bump = torch.exp(-0.5 * (r_bins / float(dc_sigma_bins)) ** 2)
    return 1.0 - dc_bump


def compute_log_rfft_spectrum(residual: np.ndarray, dc_sigma_bins: float) -> np.ndarray:
    """Native-resolution log |rFFT| spectrum with DC suppression.

    Pipeline (identical to the CNN route): rFFT2 -> vertical fftshift -> suppress
    DC on the **native complex spectrum** -> ``log1p(abs(F))``. The observed patch
    is o x o, so the output keeps its native rFFT shape ``(1, o, o//2+1)``.
    """
    x = torch.from_numpy(residual.astype(np.float32))
    if x.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {tuple(x.shape)}")
    height, width = int(x.shape[-2]), int(x.shape[-1])

    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    dc_weight = build_dc_weight(height, width, dc_sigma_bins, device=f.device, dtype=f.real.dtype)
    f = f * dc_weight
    s = torch.log1p(torch.abs(f))
    return s.unsqueeze(0).numpy().astype(np.float32)
