"""Candidate original-size estimation from resampling NFA curves.

The detector can reveal periodic spectral distances, but one distance only
identifies an original size modulo the current size. This module turns those
distances into candidate original sizes and gives each candidate an interpretable
score from the observed NFA curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resampling_core import DetectionResult


@dataclass(frozen=True)
class SizeCandidate:
    """One candidate original size for a single image axis."""

    size: int
    score: float
    predicted_distances: tuple[int, ...]
    supporting_distances: tuple[int, ...]
    support_scores: tuple[float, ...]


def significant_distances(
    result: DetectionResult,
    *,
    nfa_threshold: float | None = 1.0,
    top_k: int = 8,
) -> np.ndarray:
    """Return distances worth using as residues for candidate generation.

    If no distance passes ``nfa_threshold``, the best ``top_k`` distances are
    returned. This makes controlled experiments easier to analyze even when the
    detector is weak for a particular interpolation method or image.
    """

    if nfa_threshold is not None:
        selected = result.distances[result.nfa <= nfa_threshold]
        if selected.size:
            return np.unique(selected.astype(int))

    order = np.argsort(result.nfa)[:top_k]
    return np.unique(result.distances[order].astype(int))


def generate_size_candidates(
    current_size: int,
    distances: np.ndarray,
    *,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> list[int]:
    """Generate original-size candidates from observed spectral distances.

    A candidate ``N`` is kept when ``N mod current_size`` equals either an
    observed distance ``d`` or its symmetric residue ``current_size - d``.
    """

    if current_size <= 0:
        raise ValueError("current_size must be positive")

    min_size = max(1, int(np.floor(current_size * min_scale)))
    max_size = max(min_size, int(np.ceil(current_size * max_scale)))
    candidates: set[int] = set()

    for distance in distances.astype(int):
        if distance <= 0 or distance >= current_size:
            continue
        residues = {distance, current_size - distance}
        for residue in residues:
            if residue == 0:
                continue
            first_k = int(np.floor((min_size - residue) / current_size))
            last_k = int(np.ceil((max_size - residue) / current_size))
            for k in range(first_k, last_k + 1):
                size = k * current_size + residue
                if min_size <= size <= max_size:
                    candidates.add(int(size))

    return sorted(candidates)


def _distance_score(result: DetectionResult, distance: int, tolerance: int) -> tuple[int, float]:
    """Return the nearest tested distance and its positive ``-log10(NFA)`` score."""

    distances = result.distances.astype(int)
    nearest_index = int(np.argmin(np.abs(distances - distance)))
    nearest_distance = int(distances[nearest_index])
    if abs(nearest_distance - distance) > tolerance:
        return nearest_distance, 0.0
    return nearest_distance, max(0.0, float(-result.log10_nfa[nearest_index]))


def score_size_candidate(
    result: DetectionResult,
    current_size: int,
    candidate_size: int,
    *,
    tolerance: int = 1,
) -> SizeCandidate:
    """Score one candidate by how well its predicted peak locations are observed."""

    residue = candidate_size % current_size
    predicted: list[int] = []
    if residue != 0:
        predicted.append(residue)
        symmetric = current_size - residue
        if symmetric != residue:
            predicted.append(symmetric)

    supports: list[int] = []
    support_scores: list[float] = []
    for distance in predicted:
        nearest_distance, score = _distance_score(result, distance, tolerance)
        supports.append(nearest_distance)
        support_scores.append(score)

    score = float(np.sum(support_scores))
    return SizeCandidate(
        size=int(candidate_size),
        score=score,
        predicted_distances=tuple(int(d) for d in predicted),
        supporting_distances=tuple(supports),
        support_scores=tuple(support_scores),
    )


def rank_size_candidates(
    result: DetectionResult,
    current_size: int,
    *,
    nfa_threshold: float | None = 1.0,
    top_peak_k: int = 8,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    tolerance: int = 1,
) -> list[SizeCandidate]:
    """Generate and rank candidate original sizes for one axis."""

    distances = significant_distances(
        result,
        nfa_threshold=nfa_threshold,
        top_k=top_peak_k,
    )
    sizes = generate_size_candidates(
        current_size,
        distances,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    candidates = [
        score_size_candidate(result, current_size, size, tolerance=tolerance)
        for size in sizes
    ]
    return sorted(candidates, key=lambda item: (-item.score, item.size))


def candidate_rank(candidates: list[SizeCandidate], true_size: int) -> int | None:
    """Return the 1-based rank of ``true_size``, or ``None`` if absent."""

    for index, candidate in enumerate(candidates, start=1):
        if candidate.size == true_size:
            return index
    return None
