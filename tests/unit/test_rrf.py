import pytest

from rag.search.retrieval import rrf

W = {"dense": 1.0, "sparse": 1.0}


def test_appearing_in_both_lists_beats_rank_1_in_one():
    fused = rrf({"dense": ["a", "b", "c"], "sparse": ["d", "b", "e"]}, k=60, weights=W)
    assert fused[0][0] == "b"  # 2/62 > 1/61
    assert [i for i, _ in fused[1:3]] == ["a", "d"]


def test_scores_match_formula():
    fused = dict(rrf({"dense": ["a", "b"], "sparse": ["b"]}, k=60, weights=W))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_weights_scale_each_list():
    fused = rrf({"dense": ["a"], "sparse": ["b"]}, k=60, weights={"dense": 1.0, "sparse": 2.0})
    assert fused == [("b", pytest.approx(2 / 61)), ("a", pytest.approx(1 / 61))]


def test_ties_are_deterministic():
    fused = rrf({"dense": ["b"], "sparse": ["a"]}, k=60, weights=W)
    assert [i for i, _ in fused] == ["a", "b"]


def test_empty_lists():
    assert rrf({"dense": [], "sparse": []}, k=60, weights=W) == []
