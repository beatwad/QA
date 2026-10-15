from rag.eval.runner import grade, sample, share, top_ce

from fakes import live_hit


def payload(pids, lang):
    return {"paragraph_ids": pids, "lang": lang}


def test_grade_same_language_parallel_and_copies():
    assert grade(payload(["art03-p00"], "ru"), "art03-p00", "ru") == 2
    assert grade(payload(["art03-p00"], "en"), "art03-p00", "ru") == 1   # parallel paragraph
    assert grade(payload(["art03-p00-old"], "ru"), "art03-p00", "ru") == 0  # archived copy is not gold
    assert grade(payload(["art03-p01"], "ru"), "art03-p00", "ru") == 0


def test_top_ce_is_what_the_gate_sees():
    a, b = live_hit(1), live_hit(2)
    a.ce_score, b.ce_score = -1.0, 3.0
    assert top_ce([a, b]) == 3.0
    assert top_ce([]) == float("-inf")  # nothing survived post-ranking -> always blocked


def test_share():
    assert share([1.0, -1.0, 2.0], lambda s: s >= 0) == 2 / 3


def test_sample_keeps_other_kinds_and_is_deterministic():
    golden = ([{"kind": "answerable", "lang": lang, "id": f"{lang}{i}"} for lang in ("en", "ru") for i in range(10)]
              + [{"kind": "unanswerable", "lang": "en", "id": "u"}])
    picked = sample(golden, limit=3)
    assert sum(g["kind"] == "answerable" and g["lang"] == "en" for g in picked) == 3
    assert sum(g["kind"] == "answerable" and g["lang"] == "ru" for g in picked) == 3
    assert any(g["kind"] == "unanswerable" for g in picked)
    assert [g["id"] for g in picked] == [g["id"] for g in sample(golden, limit=3)]
    assert sample(golden, limit=None) == golden
