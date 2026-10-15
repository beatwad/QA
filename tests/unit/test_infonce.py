import math

import pytest
import torch

from rag.train.loss import InfoNCE

# two queries, two chunks, unit vectors in 2D
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
C = torch.tensor([[0.6, 0.8], [0.8, 0.6]])


def by_hand(q, c, t, hard=()):
    """The design.md formula with plain floats."""
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    n, chunks = len(q), [*c, *hard]
    q2c = [-math.log(math.exp(dot(q[i], c[i]) / t) / sum(math.exp(dot(q[i], k) / t) for k in chunks))
           for i in range(n)]
    c2q = [-math.log(math.exp(dot(c[i], q[i]) / t) / sum(math.exp(dot(c[i], k) / t) for k in q))
           for i in range(n)]
    return (sum(q2c) + sum(c2q)) / (2 * n)


def test_matches_the_formula():
    loss = InfoNCE(init_temperature=0.5)
    assert loss(Q, C).item() == pytest.approx(by_hand(Q.tolist(), C.tolist(), 0.5), rel=1e-5)


def test_hard_negatives_join_the_query_side_denominator():
    hard = torch.tensor([[0.0, -1.0]])
    loss = InfoNCE(init_temperature=0.5)
    assert loss(Q, C, hard).item() == pytest.approx(by_hand(Q.tolist(), C.tolist(), 0.5, hard.tolist()), rel=1e-5)
    assert loss(Q, C, hard).item() > loss(Q, C).item()


def test_symmetric_without_hard_negatives():
    loss = InfoNCE()
    assert loss(Q, C).item() == pytest.approx(loss(C, Q).item())


def test_positive_is_in_the_denominator():
    # q_i = c_i, orthogonal otherwise: sim 1 for the positive, 0 for the negative, t = 0.1 ->
    # -log(e^10 / (e^10 + e^0)) = log(1 + e^-10) > 0; without the positive in the denominator it would be -10
    perfect = InfoNCE(init_temperature=0.1)(Q, Q).item()
    assert perfect == pytest.approx(math.log(1 + math.exp(-10)), rel=1e-3)
    # one pair in the batch: the softmax is over a single item, so the loss is exactly 0
    assert InfoNCE()(Q[:1], C[:1]).item() == pytest.approx(0.0, abs=1e-7)


def test_temperature_is_clamped_and_learnable():
    loss = InfoNCE(init_temperature=0.07, min_temperature=0.05)
    with torch.no_grad():
        loss.log_temp.fill_(math.log(1e-4))
    assert loss.temperature().item() == pytest.approx(0.05)
    fresh = InfoNCE()
    fresh(Q, C).backward()
    assert fresh.log_temp.grad is not None and fresh.log_temp.grad != 0
