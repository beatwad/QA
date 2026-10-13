"""Post-ranking: business signals on top of the cross-encoder, then dedup + per-doc cap.

final = sigmoid(ce) + w_f * freshness + w_a * authority - w_d * dup
The business weights are small on purpose: they reorder near-ties but can't lift an off-topic chunk.
"""

import math
from collections import Counter
from datetime import date

import numpy as np

from rag.config import PostrankConfig
from rag.retrieval import Hit

def freshness(updated_at: date, today: date, half_life_days: float) -> float:
    """Exponential decay: 1.0 today, 0.5 after one half-life."""
    age = max((today - updated_at).days, 0)
    return 0.5 ** (age / half_life_days)


def is_live(payload: dict) -> bool:
    """Lifecycle: archived / deprecated / superseded chunks are excluded, not just demoted."""
    return payload.get("status") == "active" and not payload.get("superseded_by")


def postrank(hits: list[Hit], cfg: PostrankConfig, today: date) -> list[Hit]:
    live = [h for h in hits if is_live(h.payload)]
    if not live:
        return []
    base = []
    for h in live:
        p = h.payload
        fresh = freshness(date.fromisoformat(p["updated_at"]), today, cfg.half_life_days[p["source"]])
        authority = cfg.authority_prior[p["authority"]]
        base.append(1 / (1 + math.exp(-h.ce_score)) + cfg.w_freshness * fresh + cfg.w_authority * authority)

    vectors = np.asarray([h.vector for h in live])  # L2-normalized bge-m3 vectors from the index
    sims = vectors @ vectors.T

    # MMR-style greedy selection: each pick is penalized by its similarity to what's already picked
    selected: list[int] = []
    per_doc: Counter = Counter()
    remaining = list(range(len(live)))

    def dup(i):
        return float(max((sims[i][j] for j in selected), default=0.0))

    while remaining and len(selected) < cfg.final_k:
        best = max(remaining, key=lambda i: base[i] - cfg.w_dup * dup(i))
        remaining.remove(best)
        doc_id = live[best].payload["doc_id"]
        if dup(best) >= cfg.dup_threshold or per_doc[doc_id] >= cfg.max_per_doc:
            continue
        live[best].final_score = base[best] - cfg.w_dup * dup(best)
        selected.append(best)
        per_doc[doc_id] += 1
    return [live[i] for i in selected]
