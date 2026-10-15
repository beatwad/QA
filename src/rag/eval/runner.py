"""Offline eval on the golden set: ranking metrics per stage and per language pair, plus gate calibration.

Relevance grades for a question with gold paragraph P asked in language L:
  2 = chunk contains P and is in L;  1 = chunk contains P in the other language (parallel);  0 = anything else.
Archived/duplicate/Slack copies carry their own paragraph ids, so they always grade 0.
"""

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from loguru import logger

from rag.config import Config
from rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag.search.postrank import postrank
from rag.search.rerank import Reranker, rerank
from rag.search.retrieval import Hit, Retriever
from rag.storage.index import DENSE, SPARSE

STAGES = ("dense", "sparse", "rrf", "rerank", "postrank")
METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")
# (name, question language, languages of the chunks searched; None = whole index, as served)
SETTINGS = [
    ("EN→all", "en", None), ("RU→all", "ru", None),
    ("EN→EN", "en", ["en"]), ("RU→RU", "ru", ["ru"]),
    ("EN→RU", "en", ["ru"]), ("RU→EN", "ru", ["en"]),
]
GATE_THRESHOLDS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)


def grade(payload: dict, gold: str, lang: str) -> int:
    """If chunk contains the paragraph that answers the golden dataset's question in the question's language - return 2.
    If it contains the paragraph in a different language - 1.
    If chunk doesn't contain the paragraph - 0."""
    if gold not in payload["paragraph_ids"]:
        return 0
    return 2 if payload["lang"] == lang else 1


def scores(grades: list[int], ideal: list[int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(grades, 5), 
        "recall@10": recall_at_k(grades, 10), 
        "mrr": reciprocal_rank(grades), 
        "ndcg@10": ndcg_at_k(grades, ideal, 10)
        }


@dataclass
class Evaluator:
    cfg: Config
    retriever: Retriever
    reranker: Reranker
    users: dict[str, list[str]]
    today: date

    def __post_init__(self):
        # paragraph id -> languages of the indexed chunks containing it (one entry per chunk)
        self.chunk_langs: dict[str, list[str]] = defaultdict(list)
        index, offset = self.retriever.index, None
        while True:
            points, offset = index.client.scroll(index.collection, limit=1000, offset=offset,
                                                 with_payload=["paragraph_ids", "lang"], with_vectors=False)
            for p in points:
                for pid in p.payload["paragraph_ids"]:
                    self.chunk_langs[pid].append(p.payload["lang"])
            if offset is None:
                break

    def stages(self, question: str, groups: list[str], langs: list[str] | None) -> dict[str, list[Hit]]:
        """The ranked list after every stage of the serving pipeline."""
        fused = self.retriever.search(question, groups, langs)
        reranked = rerank(self.reranker, question, fused, self.cfg.rerank.top_k)
        return {
            "dense": self.retriever.search_branch(DENSE, question, groups, langs),
            "sparse": self.retriever.search_branch(SPARSE, question, groups, langs),
            "rrf": fused,
            "rerank": reranked,
            "postrank": postrank(reranked, self.cfg.postrank, self.today),
        }

    def retrieval(self, golden: list[dict], settings=SETTINGS) -> tuple[dict, dict[str, list[float]]]:
        """Mean metric per setting and stage; also each question's gate score in the whole-index settings."""
        results, gate_scores = {}, defaultdict(list)
        for name, q_lang, langs in settings:
            # keep only questions that have answers in the index
            questions = [g for g in golden if g["kind"] == "answerable" and g["lang"] == q_lang]
            # hold running totals of each metric, kept separately for each pipeline stage, 
            # across all questions in the current setting
            sums = {s: defaultdict(float) for s in STAGES}
            for i, g in enumerate(questions, start=1):
                # best possible ranking, same-language chunk from golden dataset gets grade 2, else grade 1
                ideal = [2 if lang == q_lang else 1 for lang in self.chunk_langs[g["gold_paragraph"]]
                         if langs is None or lang in langs]
                # call each stage (dense, sparse, rrf, rerank, postrank) for each user and language setting
                # it returns ranked list of all found chunks after each pipeline stage 
                lists = self.stages(g["question"], self.users[g["user"]], langs)
                for stage, hits in lists.items():
                    # calculate metrics for each stage
                    for metric, value in scores([grade(h.payload, g["gold_paragraph"], q_lang) for h in hits],
                                                ideal).items():
                        # we need sums to get average metric value for each stage
                        sums[stage][metric] += value
                # we need gate scores for the whole unfiltered index, so collect them
                # only when language setting is None
                if langs is None:
                    # collect gate scores in that case
                    gate_scores["answerable"].append(top_ce(lists["postrank"]))
                if i % 200 == 0:
                    logger.info("{}: {}/{} questions", name, i, len(questions))
            # return average metric for each stage
            results[name] = {s: {m: sums[s][m] / len(questions) for m in METRICS} for s in STAGES}
            results[name]["n"] = len(questions)
        return results, gate_scores

    def gate(self, golden: list[dict], gate_scores: dict[str, list[float]]) -> dict:
        """Share of questions on the right side of each candidate gate threshold.
        Is used to find the right threshold that minimizes FPs and FNs"""
        for kind in ("unanswerable", "acl"):
            for g in (g for g in golden if g["kind"] == kind):
                lists = self.stages(g["question"], self.users[g["user"]], None)
                gate_scores[kind].append(top_ce(lists["postrank"]))
        rows = {}
        for t in GATE_THRESHOLDS:
            rows[t] = {
                "answerable passed": share(gate_scores["answerable"], lambda s: s >= t),
                "unanswerable blocked": share(gate_scores["unanswerable"], lambda s: s < t),
                "acl blocked": share(gate_scores["acl"], lambda s: s < t),
            }
        summary = {k: {"n": len(v), "median": statistics.median(v), "max": max(v)} for k, v in gate_scores.items() if v}
        return {"thresholds": rows, "scores": summary}


def top_ce(final: list[Hit]) -> float:
    """What the gate looks at: the best raw cross-encoder score among the chunks that survived post-ranking."""
    return max((h.ce_score for h in final), default=float("-inf"))


def share(values: list[float], ok) -> float:
    return sum(map(ok, values)) / len(values) if values else float("nan")


def sample(golden: list[dict], limit: int | None, seed: int = 0) -> list[dict]:
    """Limits answerable questions per language (the same ones for every setting); to speed up the
    measure; other kinds (Unanswerable and ACL) are not limited because their number is small."""
    if limit is None:
        return golden
    rng = random.Random(seed)
    out = [g for g in golden if g["kind"] != "answerable"]
    for lang in ("en", "ru"):
        pool = [g for g in golden if g["kind"] == "answerable" and g["lang"] == lang]
        out += rng.sample(pool, min(limit, len(pool)))
    return out


def format_report(retrieval: dict, gate: dict, threshold: float) -> str:
    lines = []
    for name, by_stage in retrieval.items():
        lines.append(f"\n{name}  ({by_stage['n']} questions)")
        lines.append(f"  {'stage':10}" + "".join(f"{m:>11}" for m in METRICS))
        for stage in STAGES:
            lines.append(f"  {stage:10}" + "".join(f"{by_stage[stage][m]:>11.3f}" for m in METRICS))
    lines.append("\nGate (top raw cross-encoder score after post-ranking, whole index)")
    for kind, s in gate["scores"].items():
        lines.append(f"  {kind:13} n={s['n']:<5} median {s['median']:6.2f}   max {s['max']:6.2f}")
    cols = ("answerable passed", "unanswerable blocked", "acl blocked")
    lines.append(f"  {'threshold':>10}" + "".join(f"{c:>22}" for c in cols))
    for t, row in gate["thresholds"].items():
        mark = "  <- current" if t == threshold else ""
        lines.append(f"  {t:>10.1f}" + "".join(f"{row[c]:>22.3f}" for c in cols) + mark)
    return "\n".join(lines)
