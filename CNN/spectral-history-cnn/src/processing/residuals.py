from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle


def tv_residual(y: np.ndarray, weight: float, eps: float, max_num_iter: int) -> np.ndarray:
    denoised = denoise_tv_chambolle(
        y.astype(np.float32),
        weight=float(weight),
        eps=float(eps),
        max_num_iter=int(max_num_iter),
        channel_axis=None,
    )
    residual = y.astype(np.float32) - denoised.astype(np.float32)
    return residual.astype(np.float32)
