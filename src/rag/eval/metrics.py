"""Ranking metrics over graded relevance.

Each function takes `grades`: the relevance grade of every returned item, in rank order
(0 = not relevant). Averaging over questions is left to the caller.
"""

import math


def recall_at_k(grades: list[int], k: int) -> float:
    """1.0 if any relevant item is in the top k, else 0.0.

    Every question here has exactly one gold paragraph, so "did we find it" is the recall."""
    return float(any(g > 0 for g in grades[:k]))


def reciprocal_rank(grades: list[int]) -> float:
    """1 / rank of the first relevant item (rank from 1); 0.0 if none is returned. Mean over questions = MRR."""
    return next((1 / rank for rank, g in enumerate(grades, start=1) if g > 0), 0.0)


def dcg_at_k(grades: list[int], k: int) -> float:
    """Discounted Cumulative Gain: gain (2^g - 1) discounted by log2(rank + 1)."""
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades[:k], start=1))


def ndcg_at_k(grades: list[int], ideal: list[int], k: int) -> float:
    """DCG of the returned ranking divided by the DCG of the best possible one.

    `ideal` lists the grades of ALL relevant items that exist in the searched scope,
    so missing a relevant item costs as much as ranking it low."""
    best = dcg_at_k(sorted(ideal, reverse=True), k)
    return dcg_at_k(grades, k) / best if best else 0.0
