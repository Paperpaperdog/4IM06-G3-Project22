import numpy as np
import torch
import torch.nn.functional as F


def compute_log_rfft_spectrum(
    residual: np.ndarray,
    target_height: int | None,
    target_width_rfft: int | None,
    dc_sigma_bins: float,
) -> np.ndarray:
    """Log |rFFT| magnitude spectrum of a residual, with DC suppression.

    If ``target_height``/``target_width_rfft`` are given, the spectrum is
    resampled onto that common normalized frequency grid (the 512x257 Mask
    behaviour). If either is ``None`` the spectrum is kept at its **native**
    rFFT resolution ``(H, H//2+1)`` (used when each input size is trained
    separately, mirroring the CNN route).
    """
    x = torch.from_numpy(residual.astype(np.float32))
    if x.ndim != 2:
        raise ValueError(f"Expected residual shape [H,W], got {tuple(x.shape)}")

    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    s = torch.log1p(torch.abs(f))
    if target_height is not None and target_width_rfft is not None:
        s = F.interpolate(
            s[None, None], size=(target_height, target_width_rfft), mode="bilinear", align_corners=True
        )[0, 0]

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
