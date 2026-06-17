import numpy as np
from skimage.restoration import denoise_tv_chambolle


def tv_residual(y: np.ndarray, weight: float, max_num_iter: int, eps: float = 1e-4) -> np.ndarray:
    denoised = denoise_tv_chambolle(
        y.astype(np.float32),
        weight=weight,
        eps=float(eps),
        max_num_iter=max_num_iter,
        channel_axis=None,
    )
    return (y.astype(np.float32) - denoised.astype(np.float32)).astype(np.float32)
