import numpy as np
from skimage.restoration import denoise_tv_chambolle


def tv_residual(y: np.ndarray, weight: float, max_num_iter: int) -> np.ndarray:
    denoised = denoise_tv_chambolle(
        y,
        weight=weight,
        eps=1e-4,
        max_num_iter=max_num_iter,
        channel_axis=None,
    )
    return (y - denoised).astype(np.float32)
