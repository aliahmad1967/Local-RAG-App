"""Reciprocal Rank Fusion is pure and needs no database."""

from __future__ import annotations

import pytest

from localrag.retrieval.fusion import rank_map, reciprocal_rank_fusion, score_map


def test_rrf_rewards_agreement_between_lists() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", -1.2), ("a", -2.0), ("d", -3.0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    assert [chunk_id for chunk_id, _ in fused[:2]] == ["a", "b"]
    assert dict(fused)["a"] > dict(fused)["d"]
    assert dict(fused)["a"] > 1 / 61  # appears in both lists


def test_rrf_respects_weights_and_is_stable() -> None:
    first = [("x", 1.0), ("y", 1.0)]
    second = [("y", 1.0), ("x", 1.0)]

    weighted = reciprocal_rank_fusion([first, second], k=0, weights=[10.0, 1.0])
    assert [chunk_id for chunk_id, _ in weighted] == ["x", "y"]
    assert dict(weighted)["x"] == pytest.approx(10.5)
    assert dict(weighted)["y"] == pytest.approx(6.0)

    unweighted = reciprocal_rank_fusion([first, second], k=0)
    assert [chunk_id for chunk_id, _ in unweighted] == ["x", "y"]


def test_rrf_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1.0)]], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1.0)]], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("", 1.0)]])


def test_rank_and_score_maps() -> None:
    ranking = [("a", 0.5), ("b", 0.25)]
    assert rank_map(ranking) == {"a": 1, "b": 2}
    assert score_map(ranking) == {"a": 0.5, "b": 0.25}
