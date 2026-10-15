import random

import numpy as np

from rag.train.data import QAPair, mine_negatives, one_per_paragraph

# paragraph -> axis; the fake model sees a question as close to every paragraph it names
PARAGRAPHS = ["Tesla died in 1943.", "Tesla was born in Smiljan.", "Edison died in 1931.", "Warsaw is in Poland."]


def encode(texts):
    vecs = np.array([[1.0 if p.split()[0] in t or p in t else 0.0 for p in PARAGRAPHS] for t in texts])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_one_question_per_paragraph():
    pairs = [QAPair(f"q{i}", PARAGRAPHS[i % 2], []) for i in range(6)]
    picked = one_per_paragraph(pairs, limit=10, rng=random.Random(0))
    assert sorted(p.paragraph for p in picked) == sorted(PARAGRAPHS[:2])
    assert len(one_per_paragraph(pairs, limit=1, rng=random.Random(0))) == 1


def test_mined_negative_is_close_but_not_gold_and_has_no_answer():
    pair = QAPair("When did Tesla die?", PARAGRAPHS[0], ["1943"])
    for seed in range(5):
        [ex] = mine_negatives([pair], PARAGRAPHS, encode, top_k=2, rng=random.Random(seed))
        assert ex.negative == "Tesla was born in Smiljan."  # top-2 = both Tesla paragraphs, minus the gold one


def test_paragraph_containing_the_answer_is_not_a_negative():
    pair = QAPair("When did Tesla die? 1943", PARAGRAPHS[0], ["Tesla"])  # every Tesla paragraph has the answer
    assert mine_negatives([pair], PARAGRAPHS, encode, top_k=2, rng=random.Random(0)) == []
