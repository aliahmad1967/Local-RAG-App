"""Reciprocal Rank Fusion (RRF).

RRF combines ranked lists without needing comparable scores: a document's fused
score is the sum of `weight / (k + rank)` over every list it appears in. Lower
ranks (better results) contribute more. The implementation is deliberately a
pure function so it is easy to test and reuse in the evaluation script.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse any number of ranked ``[(chunk_id, score), ...]`` lists.

    Args:
        rankings: Already-sorted candidate lists, best first. Their raw scores
            are ignored by RRF but preserved by the caller for the debug panel.
        k: RRF smoothing constant (classically 60).
        weights: Optional per-list weight, e.g. ``[1.0, 1.0]``. Use it to favour
            BM25 for exact Arabic names or dense for conceptual queries.

    Returns:
        ``[(chunk_id, fused_score), ...]`` sorted descending by fused score.
        Ties keep the order in which the ids were first encountered (stable).
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    effective_weights = list(weights) if weights is not None else [1.0] * len(rankings)

    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking_index, ranking in enumerate(rankings):
        weight = effective_weights[ranking_index]
        if weight == 0:
            continue
        for position, (chunk_id, _score) in enumerate(ranking, start=1):
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("RRF chunk ids must be non-empty strings")
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + position)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = order
                order += 1

    return sorted(fused.items(), key=lambda item: (-item[1], first_seen[item[0]]))


def rank_map(ranking: Iterable[tuple[str, float]]) -> dict[str, int]:
    """Map chunk ids to their 1-based rank in a single ranked list."""
    return {chunk_id: rank for rank, (chunk_id, _score) in enumerate(ranking, start=1)}


def score_map(ranking: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Map chunk ids to their raw score in a single ranked list."""
    return {chunk_id: float(score) for chunk_id, score in ranking}
