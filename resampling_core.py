"""Minimal resampling detection core from spectral correlation + NFA.

This is a clean-room, simple implementation of the main ideas in
``resampling_detection (1).pdf``:

1. extract a residual with TV denoising;
2. compute complex Pearson correlations between Fourier-spectrum patches;
3. convert local correlation maxima counts into an a-contrario NFA.

The implementation favors readability over speed. The paper's optimized
version uses integral images and FFTs to accelerate the same quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_float


EPS = 1e-12


@dataclass(frozen=True)
class DetectionResult: #这是输出结果的类，包含：
    """Result for one detection axis."""

    axis: int #检测方向，0表示垂直方向，1表示水平方向
    patch_shape: tuple[int, int] #使用的patch的形状，(h, w)，表示每个patch的高度和宽度
    radius: int #局部最大窗口半径r
    distances: np.ndarray #测试的距离d数组
    correlations: np.ndarray #每个patch，每个距离d的相关性值，形状为(patch_rows, patch_cols, num_distances)
    maxima_counts: np.ndarray #每个patch的局部最大值计数
    nfa: np.ndarray #每个d的NFA值
    log10_nfa: np.ndarray #每个d的log10(NFA)值

#     这几个数组是一一对应的。比如：
#   i = result.best_index
#   result.distances[i]
#   result.nfa[i]
#     表示最显著的频谱相关距离和它的 NFA。

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.nfa))

    @property
    def best_distance(self) -> int:
        return int(self.distances[self.best_index])

    @property
    def best_nfa(self) -> float:
        return float(self.nfa[self.best_index])


def as_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert an input image to a 2D float array in [0, 1].
    把输入图像统一成二维灰度 float 图像，范围在 [0, 1]。如果输入是 RGB 图像，取前三个通道并转换为灰度图。如果输入已经是二维的，就直接转换为 float。
    """

    image = img_as_float(image) # skimage.util.img_as_float(image) 会根据原始数据类型做规范转换。对于 uint8 图像，它会把 [0,255] 转成 [0,1]
    if image.ndim == 3:
        image = rgb2gray(image[..., :3]) # 如果输入是 RGB 图像，取前三个通道并转换为灰度图。
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")
    return image.astype(np.float64, copy=False) # 如果输入已经是二维的，就直接转换为 float （从 uint8 变成了 float64）。


def tv_residual(image: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Return image - TV_denoised(image).

    In the paper's terminology, TV denoising estimates the bounded-variation
    component: piecewise smooth content and strong edges. Subtracting it leaves
    a residual containing noise and small oscillations, including resampling
    traces.
    初步实现：用TV denoising得到残差图像 （估计原图中的平滑部分和强边缘，然后用原图减去这个平滑部分）得到残差图像。
    残差图像包含了噪声和小的振荡，包括重采样的痕迹。
    残差的目的是，剔除自然图像中本身的结构性相关，避免误检。
    """

    gray = as_gray_float(image)
    denoised = denoise_tv_chambolle(gray, weight=weight, channel_axis=None)
    return gray - denoised


def residual_spectrum(residual: np.ndarray) -> np.ndarray:
    """Compute the centered 2D Fourier spectrum of a residual image.
    对residual图像做Fourier transform，得到频谱图。使用fft2计算二维FFT，fftshift把零频分量移到中心位置。

    DC 分量就是 Fourier 频谱里的零频率分量，也可以理解为图像的整体平均亮度 / 平均值部分。
    对于一维信号，DC 分量对应频率 f=0，表示“不变化的常数部分”。对于二维图像，DC 分量对应频率坐标 (u,v)=(0,0)，表示整张图像中不随空间位置变化的平均值。
    在二维 DFT 里，零频分量本质上等于所有像素值的和：如果写成平均值，就是：F(0,0)=HW⋅mean(image)
    所以如果一张 residual 图像整体有一个非零均值，比如大多数像素都偏正一点，那么它的 DC 分量会很大。DC 分量很大，会在频谱里形成一个很大的中心亮点，干扰分析。

    np.fft.fft2(residual) 默认输出的频谱中，零频率分量，也就是 DC 分量，放在数组左上角 [0,0]。频率排列大概是这样的：

        DC / 低频        高频
        高频             负频率部分
    因此使用 np.fft.fftshift 可以把频谱中的零频分量移动到中心位置，让观察频谱图更直观

    注意，是复数数组，因为 Fourier 频谱包含幅度和相位信息。后续的相关性计算会用到复数的乘积和共轭。
    """

    residual = residual - np.mean(residual) # 去掉残差图像的均值，避免DC分量过大影响后续的相关性计算。
    return np.fft.fftshift(np.fft.fft2(residual))


def complex_patch_correlation(anchor: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    """Complex Pearson correlation magnitude for patch arrays.

    ``anchor`` and ``shifted`` must have shape ``(..., h, w)``. The last two
    axes are flattened as one patch vector. Eq. (16) in the paper defines the
    complex Pearson coefficient. For local maxima and NFA we need an ordered
    real score, so this function returns its magnitude.

    计算复数 patch 的 Pearson correlation。

    论文 Eq. (16) 是 corr(x,y) = <x - mean(x), y - mean(y)> / (||x - mean(x)|| ||y - mean(y)||)
    这里 x 和 y 是复数频谱 patch。

    输入形状：前面的 ... 可以是一批 patch，比如 (patch_rows, patch_cols, h, w)。
    anchor.shape  = (..., h, w)
    shifted.shape = (..., h, w)

    输出形状：correlation.shape = anchor.shape[:-2] 比如输入 (10, 10, 38, 38)，返回 (10, 10)。
    """

    x = anchor.reshape(anchor.shape[:-2] + (-1,)) # 把每个patch拉平成向量
    y = shifted.reshape(shifted.shape[:-2] + (-1,))

    x_centered = x - np.mean(x, axis=-1, keepdims=True) # 对每个patch的向量，减去它的均值
    y_centered = y - np.mean(y, axis=-1, keepdims=True)

    numerator = np.sum(x_centered * np.conj(y_centered), axis=-1) #复数内积，分子
    x_norm = np.sqrt(np.sum(np.abs(x_centered) ** 2, axis=-1)) #复数模，分母
    y_norm = np.sqrt(np.sum(np.abs(y_centered) ** 2, axis=-1)) #复数模，分母
    return np.abs(numerator) / np.maximum(x_norm * y_norm, EPS) #返回 correlation 的模。取模是因为复数 correlation 本身有相位，但 NFA 需要比较“强弱”，所以用实数强度排序。


def non_overlapping_patches(array: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    """Split the top-left crop of ``array`` into non-overlapping patches.
    把二维数组切成不重叠 patch，只取能完整切出的 patch 数。边缘多出来的像素会被裁掉。
    最后返回：patches.shape = (patch_rows, patch_cols, h, w)
    """

    h, w = patch_shape
    if h <= 0 or w <= 0:
        raise ValueError("patch dimensions must be positive")

    height, width = array.shape
    patch_rows = height // h
    patch_cols = width // w
    if patch_rows == 0 or patch_cols == 0:
        raise ValueError(
            f"patch shape {patch_shape} is too large for spectrum shape {array.shape}"
        )

    cropped = array[: patch_rows * h, : patch_cols * w]
    patches = cropped.reshape(patch_rows, h, patch_cols, w)
    return patches.swapaxes(1, 2)


def spectral_patch_correlations(
    spectrum: np.ndarray,
    patch_shape: tuple[int, int] | None = None,
    axis: int = 0,
) -> np.ndarray:
    """Compute rho(patch, distance) for all non-overlapping patches.

    Returns an array with shape ``(num_patch_rows, num_patch_cols, axis_length)``.
    If ``axis == 0``, distance ``d`` compares each patch with the spectrum patch
    shifted by ``d`` rows. If ``axis == 1``, it compares with a shift by ``d``
    columns. Shifts are circular because the DFT is periodic.
    计算每个 patch 在每个距离 d 上的频谱相关性。

    输入：
        spectrum: complex array, shape = (H, W)
        patch_shape: (h, w)，如果不传，默认 H/10 和 W/10
        axis: 0 或 1
        axis=0：检测竖直方向，也就是 row shift。
        axis=1：检测水平方向，也就是 column shift。
    """

    if spectrum.ndim != 2:
        raise ValueError("spectrum must be a 2D array")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    height, width = spectrum.shape
    if patch_shape is None:
        patch_shape = (max(2, height // 10), max(2, width // 10))

    anchors = non_overlapping_patches(spectrum, patch_shape) #调用前面的切patch函数，得到 anchors.shape = (patch_rows, patch_cols, h, w)
    axis_length = spectrum.shape[axis]
    correlations = np.empty(anchors.shape[:2] + (axis_length,), dtype=np.float64)

    for distance in range(axis_length): #遍历所有距离，把频谱向某方向平移d个位置，得到shifted_spectrum，然后计算和原先自身的相关度
        shifted_spectrum = np.roll(spectrum, shift=-distance, axis=axis)
        shifted = non_overlapping_patches(shifted_spectrum, patch_shape)
        correlations[..., distance] = complex_patch_correlation(anchors, shifted) #调用前面的pierson相关性函数

    return correlations


def nfa_from_correlations(correlations: np.ndarray, radius: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a-contrario NFA from patch correlations.

    For each tested distance d, count how many patches have rho(d) equal to the
    maximum in the local window [d-r, d+r]. Under H0, each patch has probability
    1/(2r+1) to peak at d, so the count follows a binomial distribution.
    把 correlation 转成 NFA。
    """

    if correlations.ndim != 3:
        raise ValueError("correlations must have shape (patch_rows, patch_cols, distances)")

    num_distances = correlations.shape[-1]
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if num_distances <= 2 * radius + 1:
        raise ValueError(
            f"radius {radius} is too large for {num_distances} tested distances"
        )

    distances = np.arange(radius + 1, num_distances - radius, dtype=int) #不测太靠近边界的距离，因为要看 [d-r, d+r] 的局部窗口。
    num_patches = int(np.prod(correlations.shape[:2]))
    probability = 1.0 / (2 * radius + 1) #零假设下，一个 patch 的局部最大落在窗口中心 d 的概率
    num_tests = len(distances)

    maxima_counts = np.empty(len(distances), dtype=int)
    nfa = np.empty(len(distances), dtype=np.float64)

    for index, distance in enumerate(distances): #遍历每个距离
        local = correlations[..., distance - radius : distance + radius + 1]
        is_local_max = correlations[..., distance] >= np.max(local, axis=-1)
        count = int(np.count_nonzero(is_local_max)) # 统计 k(d) = 有多少 patch 在 d 处取得局部最大
        maxima_counts[index] = count
        nfa[index] = num_tests * binom.sf(count - 1, num_patches, probability) #二项分布尾概率，乘以 num_tests 是因为我们测试了很多个距离，要控制多重比较下的 false alarms

    log10_nfa = np.log10(np.maximum(nfa, np.finfo(float).tiny))
    return distances, maxima_counts, nfa, log10_nfa


def detect_axis(
    image: np.ndarray,
    axis: int = 0,
    tv_weight: float = 1.0,
    patch_shape: tuple[int, int] | None = None,
    radius: int = 20,
) -> DetectionResult:
    """Run the minimal detector on one axis.
    对一张图的一条方向完整跑检测。
    单轴测试入口
    """

    residual = tv_residual(image, weight=tv_weight)
    spectrum = residual_spectrum(residual)
    correlations = spectral_patch_correlations(spectrum, patch_shape=patch_shape, axis=axis)
    distances, maxima_counts, nfa, log10_nfa = nfa_from_correlations(
        correlations, radius=radius
    )

    actual_patch_shape = patch_shape
    if actual_patch_shape is None:
        actual_patch_shape = (max(2, image.shape[0] // 10), max(2, image.shape[1] // 10))

    return DetectionResult(
        axis=axis,
        patch_shape=actual_patch_shape,
        radius=radius,
        distances=distances,
        correlations=correlations,
        maxima_counts=maxima_counts,
        nfa=nfa,
        log10_nfa=log10_nfa,
    )


def detect_both_axes(
    image: np.ndarray,
    tv_weight: float = 1.0,
    patch_shape: tuple[int, int] | None = None,
    radius: int = 20,
) -> tuple[DetectionResult, DetectionResult]:
    """Run the minimal detector independently along vertical and horizontal axes.
    分别检测两个方向，返回两个 DetectionResult。
    """

    return (
        detect_axis(image, axis=0, tv_weight=tv_weight, patch_shape=patch_shape, radius=radius),
        detect_axis(image, axis=1, tv_weight=tv_weight, patch_shape=patch_shape, radius=radius),
    )
