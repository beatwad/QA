"""Symmetric InfoNCE (design.md -> "Loss function"), written out by hand.

q_i, c_i are L2-normalized; sim = dot product; i, i is the positive pair, every other pair in the batch
is a negative, and the softmax denominator runs over all of them, the positive included:

  L_q2c(i) = -log( exp(sim(q_i, c_i)/t) / sum_k exp(sim(q_i, c_k)/t) )
  L_c2q(i) = -log( exp(sim(c_i, q_i)/t) / sum_k exp(sim(c_i, q_k)/t) )
  L = 1/(2N) * sum_i (L_q2c(i) + L_c2q(i))

Hard negatives are extra chunks: they join the q -> c denominator for every query (they aren't
anyone's positive, so they have no c -> q row of their own).
"""

import math

import torch
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, init_temperature: float = 0.07, min_temperature: float = 0.01, max_temperature: float = 1.0):
        super().__init__()
        # learned in log-scale: stays positive, and equal steps scale it by equal factors
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.bounds = (math.log(min_temperature), math.log(max_temperature))

    def temperature(self) -> torch.Tensor:
        # clamped: a temperature near 0 makes the softmax one-hot and the gradients blow up
        return self.log_temp.clamp(*self.bounds).exp()

    def forward(self, queries: torch.Tensor, chunks: torch.Tensor, hard: torch.Tensor | None = None) -> torch.Tensor:
        """queries, chunks: (N, d) aligned positive pairs; hard: (M, d) extra negative chunks. All L2-normalized."""
        t = self.temperature()
        extra_chunks = chunks if hard is None else torch.cat([chunks, hard])
        log_softmax_q2c = (queries @ extra_chunks.T / t).log_softmax(dim=1)  # (N, N + M)
        log_softmax_c2q = (chunks @ queries.T / t).log_softmax(dim=1)       # (N, N)
        positives = torch.arange(len(queries), device=queries.device)
        l_q2c = -log_softmax_q2c[positives, positives]
        l_c2q = -log_softmax_c2q[positives, positives]
        return (l_q2c.sum() + l_c2q.sum()) / (2 * len(queries))
