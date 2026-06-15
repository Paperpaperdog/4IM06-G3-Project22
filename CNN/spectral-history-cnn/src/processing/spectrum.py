from __future__ import annotations

import numpy as np
import torch


def build_dc_weight(height: int, width: int, dc_sigma_bins: float, device=None, dtype=torch.float32) -> torch.Tensor:
    u = torch.fft.rfftfreq(width, d=1.0, device=device).to(dtype)
    v = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype)
    v = torch.fft.fftshift(v)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    r_bins = torch.sqrt((u_grid * width) ** 2 + (v_grid * height) ** 2 + 1e-12)
    dc_bump = torch.exp(-0.5 * (r_bins / float(dc_sigma_bins)) ** 2)
    return 1.0 - dc_bump


def compute_log_rfft_spectrum(residual: np.ndarray, dc_sigma_bins: float = 3.0) -> np.ndarray:
    if residual.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {residual.shape}.")
    height, width = residual.shape
    if height != 64 or width != 64:
        raise ValueError(f"Version 1 expects a 64x64 residual, got {height}x{width}.")

    x = torch.from_numpy(residual.astype(np.float32)).unsqueeze(0)
    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    dc_weight = build_dc_weight(height, width, dc_sigma_bins, device=f.device, dtype=f.real.dtype)
    f = f * dc_weight.unsqueeze(0)
    spectrum = torch.log1p(torch.abs(f))
    return spectrum.numpy().astype(np.float32)
