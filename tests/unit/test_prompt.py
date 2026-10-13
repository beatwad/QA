from rag.prompt import PER_MESSAGE_OVERHEAD, SYSTEM, build_messages, format_chunk, is_not_found

from fakes import chunk_hit, words

BIG = 10_000


def test_chunks_are_numbered_in_rank_order_with_source_tags():
    hits = [chunk_hit(1), chunk_hit(2), chunk_hit(3)]
    messages, used = build_messages("Q?", hits, [], words, BIG)
    user = messages[-1]["content"]
    assert used == hits
    assert user.index("[1] Title 1 (wiki, updated 2025-01-01)") < user.index("[2] Title 2") < user.index("[3] Title 3")
    assert user.endswith("Question: Q?")
    assert messages[0] == {"role": "system", "content": SYSTEM}


def test_budget_keeps_strongest_chunks_first():
    hits = [chunk_hit(i, " ".join(["w"] * 100)) for i in range(1, 6)]
    fixed = words(SYSTEM) + words("Q?") + 3 * PER_MESSAGE_OVERHEAD
    per_chunk = words(format_chunk(1, hits[0])) + 2
    _, used = build_messages("Q?", hits, [], words, budget=fixed + 3 * per_chunk)
    assert [h.chunk_id for h in used] == ["d1-c000", "d2-c000", "d3-c000"]


def test_history_keeps_most_recent_turns_and_never_drops_the_question():
    history = [{"role": "user", "content": f"turn {i} " + "w " * 20} for i in range(10)]
    messages, used = build_messages("Q?", [chunk_hit(1)], history, words, budget=200)
    turns = messages[1:-1]
    assert 0 < len(turns) < 10
    assert turns[-1]["content"].startswith("turn 9")
    assert [int(t["content"].split()[1]) for t in turns] == sorted(int(t["content"].split()[1]) for t in turns)
    assert used and messages[-1]["content"].endswith("Question: Q?")


def test_not_found_sentinel():
    assert is_not_found("NOT_FOUND")
    assert is_not_found("  NOT_FOUND\n")
    assert not is_not_found("Tesla died in 1943 [1].")
