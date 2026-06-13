import numpy as np
import torch


def compute_log_rfft_spectrum(residual: np.ndarray, dc_sigma_bins: float) -> np.ndarray:
    x = torch.from_numpy(residual.astype(np.float32))
    if x.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {tuple(x.shape)}")

    h, w = x.shape
    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)

    u = torch.fft.rfftfreq(w, d=1.0)
    v = torch.fft.fftfreq(h, d=1.0)
    v = torch.fft.fftshift(v)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    r_bins = torch.sqrt((uu * w) ** 2 + (vv * h) ** 2 + 1e-12)
    dc_bump = torch.exp(-0.5 * (r_bins / dc_sigma_bins) ** 2)
    dc_weight = 1.0 - dc_bump

    s = torch.log1p(torch.abs(f * dc_weight))
    return s.unsqueeze(0).numpy().astype(np.float32)
