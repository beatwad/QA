"""Training examples for the bi-encoder: (query, positive chunk, hard negative chunk).

- SQuAD v1.1 train (EN) and SberQuAD train (RU): (question, paragraph) pairs. The hard negative is
  mined from the current model's top-k over the same dataset's paragraphs ("hard": the model ranks it
  high, but it isn't the answer). Never the SQuAD dev split: XQuAD, the eval corpus, comes from it.
- MIRACL en/ru triplets: the negative is a passage annotators judged non-relevant.
"""

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

QA_SETS = {"squad": ("rajpurkar/squad", None), "sberquad": ("kuznetsoffandrey/sberquad", "sberquad")}
MIRACL = "sentence-transformers/miracl"

Encode = Callable[[list[str]], np.ndarray]  # texts -> (n, d) L2-normalized


@dataclass
class Example:
    query: str
    positive: str
    negative: str


@dataclass
class QAPair:
    question: str
    paragraph: str
    answers: list[str]


def load_qa(name: str) -> list[QAPair]:
    from datasets import load_dataset

    repo, config = QA_SETS[name]
    return [QAPair(r["question"], r["context"], r["answers"]["text"])
            for r in load_dataset(repo, config, split="train")]


def one_per_paragraph(pairs: list[QAPair], limit: int, rng: random.Random) -> list[QAPair]:
    """Keep one random question per paragraph, return at most `limit`. SQuAD has ~5 questions per paragraph; 
    two of them in one batch would make each other's positive an in-batch negative."""
    by_paragraph = defaultdict(list)
    for p in pairs:
        by_paragraph[p.paragraph].append(p)
    picked = [rng.choice(group) for group in by_paragraph.values()]
    rng.shuffle(picked)
    return picked[:limit]


def mine_negatives(pairs: list[QAPair], paragraphs: list[str], encode: Encode, top_k: int,
                   rng: random.Random, block: int = 1024) -> list[Example]:
    """For each pair, get a random paragraph from the model's top_k that is neither the gold one nor contains
    a reference answer (those are likely unlabeled positives, not negatives). No such paragraph -> pair dropped.
    Thus we mine hard negatives."""
    index = encode(paragraphs)
    examples = []
    for start in range(0, len(pairs), block):
        part = pairs[start : start + block]
        scores = encode([p.question for p in part]) @ index.T
        top = np.argsort(-scores, axis=1)[:, :top_k]
        for pair, candidates in zip(part, top):
            valid = [paragraphs[i] for i in candidates if paragraphs[i] != pair.paragraph
                     and not any(a in paragraphs[i] for a in pair.answers)]
            if valid:
                examples.append(Example(pair.question, pair.paragraph, rng.choice(valid)))
    return examples


def load_miracl(lang: str, limit: int, rng: random.Random) -> list[Example]:
    """Load MIRACL triplets, one triplet per query (the set repeats a query once per judged negative), 
    at most `limit`."""
    from datasets import load_dataset

    by_query = defaultdict(list)
    for r in load_dataset(MIRACL, f"{lang}-triplet", split="train"):
        by_query[r["anchor"]].append(Example(r["anchor"], r["positive"], r["negative"]))
    picked = [rng.choice(group) for group in by_query.values()]
    rng.shuffle(picked)
    return picked[:limit]


def build_examples(encode: Encode, per_source: int, top_k: int, rng: random.Random) -> list[Example]:
    """SQuAD + SberQuAD with mined negatives + MIRACL en/ru, shuffled together."""
    examples = []
    for name in QA_SETS:
        pairs = load_qa(name)
        paragraphs = sorted({p.paragraph for p in pairs})
        examples += mine_negatives(one_per_paragraph(pairs, per_source, rng), paragraphs, encode, top_k, rng)
    for lang in ("en", "ru"):
        examples += load_miracl(lang, per_source, rng)
    rng.shuffle(examples)
    return examples
