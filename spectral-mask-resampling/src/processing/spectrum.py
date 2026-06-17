import numpy as np
import torch


def compute_log_rfft_spectrum(residual: np.ndarray, dc_sigma_bins: float) -> np.ndarray:
    """Native-resolution log |rFFT| magnitude spectrum of a residual.

    The observed patch is o x o, so the spectrum keeps its native rFFT shape
    ``(o, o//2+1)`` (each input size is trained separately, mirroring the CNN
    route). DC energy is suppressed with a radial Gaussian notch.
    """
    x = torch.from_numpy(residual.astype(np.float32))
    if x.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {tuple(x.shape)}")

    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    s = torch.log1p(torch.abs(f))

    height = int(s.shape[-2])
    width_rfft = int(s.shape[-1])
    rows = torch.arange(height, dtype=torch.float32) - height // 2
    cols = torch.arange(width_rfft, dtype=torch.float32)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    r_bins = torch.sqrt(uu**2 + vv**2 + 1e-12)
    dc_bump = torch.exp(-0.5 * (r_bins / dc_sigma_bins) ** 2)
    dc_weight = 1.0 - dc_bump

    s = s * dc_weight
    return s.unsqueeze(0).numpy().astype(np.float32)
