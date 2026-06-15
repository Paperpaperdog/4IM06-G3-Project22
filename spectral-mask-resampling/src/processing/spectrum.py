import numpy as np
import torch
import torch.nn.functional as F


def compute_log_rfft_spectrum(
    residual: np.ndarray,
    target_height: int,
    target_width_rfft: int,
    dc_sigma_bins: float,
) -> np.ndarray:
    x = torch.from_numpy(residual.astype(np.float32))
    if x.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {tuple(x.shape)}")

    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    s = torch.log1p(torch.abs(f))[None, None]
    s = F.interpolate(s, size=(target_height, target_width_rfft), mode="bilinear", align_corners=True)[0, 0]

    rows = torch.arange(target_height, dtype=torch.float32) - target_height // 2
    cols = torch.arange(target_width_rfft, dtype=torch.float32)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    r_bins = torch.sqrt(uu**2 + vv**2 + 1e-12)
    dc_bump = torch.exp(-0.5 * (r_bins / dc_sigma_bins) ** 2)
    dc_weight = 1.0 - dc_bump

    s = s * dc_weight
    return s.unsqueeze(0).numpy().astype(np.float32)
