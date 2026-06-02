from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resampling_core import DetectionResult, detect_axis

from .config import (
    DETECTION_RADIUS,
    JPEG_PROBE_DIVISORS,
    LOG10_NFA_THRESHOLD,
    TARGET_SIZE,
    TV_WEIGHT,
)


def jpeg_period_distances(width: int, divisors: tuple[int, ...] = JPEG_PROBE_DIVISORS) -> list[int]:
    periods: list[int] = []
    for divisor in divisors:
        step = width // divisor
        if step > 0:
            periods.append(step)
    return sorted(set(periods))


def detect_horizontal(image: np.ndarray) -> DetectionResult:
    return detect_axis(
        image,
        axis=1,
        tv_weight=TV_WEIGHT,
        radius=DETECTION_RADIUS,
    )


def lookup_nfa(result: DetectionResult, distance: int) -> float | None:
    indices = np.where(result.distances == distance)[0]
    if len(indices) == 0:
        return None
    return float(result.nfa[int(indices[0])])


def lookup_log10_nfa(result: DetectionResult, distance: int) -> float | None:
    indices = np.where(result.distances == distance)[0]
    if len(indices) == 0:
        return None
    return float(result.log10_nfa[int(indices[0])])


def mean_correlation_profile(result: DetectionResult) -> np.ndarray:
    return np.mean(result.correlations, axis=(0, 1))


def peak_shape_features(result: DetectionResult) -> dict[str, float]:
    profile = mean_correlation_profile(result)
    peak_index = int(np.argmax(profile))
    peak_value = float(profile[peak_index])
    median = float(np.median(profile))
    prominence = peak_value - median

    half = peak_value * 0.5
    left = peak_index
    while left > 0 and profile[left] >= half:
        left -= 1
    right = peak_index
    while right < len(profile) - 1 and profile[right] >= half:
        right += 1
    width = float(right - left)

    left_side = float(np.sum(profile[max(0, peak_index - 3) : peak_index]))
    right_side = float(np.sum(profile[peak_index + 1 : peak_index + 4]))
    side_ratio = float(np.log((right_side + 1e-3) / (left_side + 1e-3)))

    nfa_best_idx = int(np.argmin(result.nfa))
    nfa_distance = int(result.distances[nfa_best_idx])

    return {
        "corr_peak_index": float(peak_index),
        "corr_peak_height": peak_value,
        "corr_prominence": prominence,
        "corr_half_width": width,
        "corr_side_ratio": side_ratio,
        "nfa_best_distance": float(nfa_distance),
        "nfa_best_log10": float(result.log10_nfa[nfa_best_idx]),
        "nfa_best_value": float(result.nfa[nfa_best_idx]),
        "is_significant": float(result.log10_nfa[nfa_best_idx] < LOG10_NFA_THRESHOLD),
    }


def probe_distance_row(result: DetectionResult, distances: tuple[int, ...]) -> dict[str, float]:
    row: dict[str, float] = {}
    for distance in distances:
        log_nfa = lookup_log10_nfa(result, distance)
        row[f"log10_nfa_d{distance}"] = log_nfa if log_nfa is not None else float("nan")
    return row


def detection_summary(image: np.ndarray, probe_distances: tuple[int, ...]) -> dict[str, float]:
    result = detect_horizontal(image)
    features = peak_shape_features(result)
    features.update(probe_distance_row(result, probe_distances))
    features["image_width"] = float(image.shape[1])
    for period in jpeg_period_distances(image.shape[1]):
        features[f"jpeg_period_{period}"] = float(period)
    return features
