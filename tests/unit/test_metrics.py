import math

import pytest

from rag.eval.metrics import dcg_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k([0, 0, 2, 0], k=3) == 1.0
    assert recall_at_k([0, 0, 2, 0], k=2) == 0.0
    assert recall_at_k([0, 1], k=5) == 1.0  # the parallel paragraph counts as found
    assert recall_at_k([], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([2, 0]) == 1.0
    assert reciprocal_rank([0, 0, 1, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0]) == 0.0


def test_dcg_by_hand():
    # rank 1: (2^2 - 1) / log2(2) = 3 ; rank 3: (2^1 - 1) / log2(4) = 0.5
    assert dcg_at_k([2, 0, 1], k=3) == pytest.approx(3.5)
    assert dcg_at_k([2, 0, 1], k=2) == pytest.approx(3.0)


def test_ndcg_by_hand():
    # ideal: [2, 1] -> 3 + 1/log2(3) ; got: [0, 1, 0, 2] -> 1/log2(3) + 3/log2(5)
    got = 1 / math.log2(3) + 3 / math.log2(5)
    best = 3 + 1 / math.log2(3)
    assert ndcg_at_k([0, 1, 0, 2], ideal=[1, 2], k=10) == pytest.approx(got / best)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k([2, 1, 0], ideal=[2, 1], k=10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0], ideal=[2], k=10) == 0.0
    assert ndcg_at_k([0], ideal=[], k=10) == 0.0


def test_ndcg_counts_relevant_items_beyond_k_as_missed():
    # the gold item exists but sits at rank 3, outside k=2
    assert ndcg_at_k([0, 0, 2], ideal=[2], k=2) == 0.0
