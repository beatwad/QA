"""Download XQuAD (EN + RU) and SQuAD v1.1 dev titles, then build data/:

- data/corpus/<doc_id>.json  one document = one article in one language
- data/corpus/meta.yaml      synthetic metadata (ACL, source, dates, lifecycle)
- data/users.yaml            user -> groups
- data/golden.jsonl          eval questions

Everything is deterministic (fixed seed).
"""

import json
import random
import re
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import yaml
from datasets import load_dataset

SEED = 42
LANGS = ("en", "ru")
DATA = Path(__file__).resolve().parents[1] / "data"

USERS = {
    "alice": ["all-employees"],
    "bob": ["all-employees", "research"],
    "hana": ["all-employees", "hr-team"],
    "root": ["all-employees", "research", "hr-team"],  # retrieval eval sees everything
    "ops": ["all-employees", "admin"],  # may call /v1/admin/sync
}

N_RESTRICTED = 4  # per group
N_SUPERSEDED = 3
N_INFORMAL = 3
N_DUPLICATE = 3
N_GENERATION = 30  # questions, asked in both languages
N_ACL = 10  # questions, asked in both languages

# (en, ru) pairs the corpus doesn't cover
UNANSWERABLE = [
    ("How many vacation days do new employees get per year?", "Сколько дней отпуска в год получают новые сотрудники?"),
    ("What is the Wi-Fi password in the Berlin office?", "Какой пароль от Wi-Fi в берлинском офисе?"),
    ("Who won the 2018 FIFA World Cup?", "Кто выиграл чемпионат мира по футболу 2018 года?"),
    ("What is the boiling point of ethanol?", "Какова температура кипения этанола?"),
    ("Who wrote the novel War and Peace?", "Кто написал роман «Война и мир»?"),
    ("How do I reset my VPN token?", "Как сбросить токен VPN?"),
    ("How tall is the Eiffel Tower?", "Какова высота Эйфелевой башни?"),
    ("What programming language is Django written in?", "На каком языке программирования написан Django?"),
    ("What is the expense limit for business dinners?", "Каков лимит расходов на деловые ужины?"),
    ("When was the first iPhone released?", "Когда вышел первый iPhone?"),
    ("What is the population of Iceland?", "Какова численность населения Исландии?"),
    ("How do I request a new laptop from IT?", "Как заказать новый ноутбук через ИТ-отдел?"),
    ("What is the official currency of Japan?", "Какая официальная валюта Японии?"),
    ("How many moons does Mars have?", "Сколько спутников у Марса?"),
    ("What is the deadline for submitting quarterly OKRs?", "Какой срок сдачи квартальных OKR?"),
]


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def year_pattern(year: str) -> re.Pattern:
    return re.compile(rf"(?<!\d){year}(?!\d)")


def load_articles():
    """Return (articles, questions).

    articles: OrderedDict art_id -> {"title", "paragraphs": OrderedDict pid -> {lang: text}}
    questions: list of {"qid", "pid", "art", lang: {"question", "answers"}}
    """
    titles = {}
    for r in load_dataset("rajpurkar/squad", split="validation"):
        titles[r["id"]] = r["title"].replace("_", " ")
    rows = {lang: load_dataset("google/xquad", f"xquad.{lang}", split="validation") for lang in LANGS}

    # Paragraph and article order follow first appearance in the EN split.
    articles: OrderedDict = OrderedDict()
    qid_to_pid = {}
    ctx_to_pid = {}
    for r in rows["en"]:
        title = titles[r["id"]]
        art_id = next((a for a, v in articles.items() if v["title"] == title), None)
        if art_id is None:
            art_id = f"art{len(articles):02d}"
            articles[art_id] = {"title": title, "paragraphs": OrderedDict()}
        paras = articles[art_id]["paragraphs"]
        if r["context"] not in ctx_to_pid:
            pid = f"{art_id}-p{len(paras):02d}"
            ctx_to_pid[r["context"]] = pid
            paras[pid] = {"en": r["context"]}
        qid_to_pid[r["id"]] = ctx_to_pid[r["context"]]

    by_qid = {qid: {"qid": qid, "pid": pid, "art": pid.split("-")[0]} for qid, pid in qid_to_pid.items()}
    for lang in LANGS:
        for r in rows[lang]:
            q = by_qid[r["id"]]
            q[lang] = {"question": r["question"], "answers": r["answers"]["text"]}
            if lang != "en":
                para = articles[q["art"]]["paragraphs"][q["pid"]]
                existing = para.setdefault(lang, r["context"])
                assert existing == r["context"], f"{r['id']}: {lang} context differs within a paragraph"
    return articles, list(by_qid.values())


def year_candidates(articles, questions):
    """Questions whose answer is a year that appears exactly once in both language versions."""
    out = []
    for q in questions:
        year = q["en"]["answers"][0]
        if not re.fullmatch(r"\d{4}", year):
            continue
        para = articles[q["art"]]["paragraphs"][q["pid"]]
        if all(len(year_pattern(year).findall(para[lang])) == 1 for lang in LANGS):
            out.append(q)
    return out


def random_date(rng, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def build(rng: random.Random):
    articles, questions = load_articles()
    art_ids = list(articles)
    shuffled = art_ids[:]
    rng.shuffle(shuffled)

    acl = {a: ["all-employees"] for a in art_ids}
    for a in shuffled[:N_RESTRICTED]:
        acl[a] = ["hr-team"]
    for a in shuffled[N_RESTRICTED : 2 * N_RESTRICTED]:
        acl[a] = ["research"]
    open_arts = shuffled[2 * N_RESTRICTED :]

    # Fact-changing edge cases need a year answer that can be swapped cleanly.
    cands = {}
    for q in year_candidates(articles, questions):
        if q["art"] in open_arts:
            cands.setdefault(q["art"], q)
    fact_arts = [a for a in open_arts if a in cands]
    assert len(fact_arts) >= N_SUPERSEDED + N_INFORMAL, "not enough year-answer articles"
    superseded = fact_arts[:N_SUPERSEDED]
    informal = fact_arts[N_SUPERSEDED : N_SUPERSEDED + N_INFORMAL]
    used = set(superseded) | set(informal)
    duplicated = [a for a in open_arts if a not in used][:N_DUPLICATE]

    changed = {}  # art -> (question, true year, wrong year)
    for a in superseded + informal:
        q = cands[a]
        year = q["en"]["answers"][0]
        para = articles[a]["paragraphs"][q["pid"]]
        while True:
            wrong = str(int(year) + rng.choice([-7, -5, -3, 3, 5, 7]))
            if not any(year_pattern(wrong).search(para[lang]) for lang in LANGS):
                break
        changed[a] = (q, year, wrong)

    docs, meta = [], []

    def add(doc_id, title, lang, paragraphs, **m):
        docs.append({"doc_id": doc_id, "title": title, "lang": lang, "paragraphs": paragraphs})
        meta.append({"doc_id": doc_id, "title": title, "lang": lang, **m})

    for a in art_ids:
        art = articles[a]
        source = rng.choice(["confluence", "gdrive", "wiki", "ticket"])
        authority = rng.choice(["official", "verified"])
        updated = random_date(rng, date(2022, 1, 1), date(2026, 6, 30))
        for lang in LANGS:
            doc_id = f"{a}-{lang}"
            paras = [{"id": pid, "text": t[lang]} for pid, t in art["paragraphs"].items()]
            add(doc_id, art["title"], lang, paras,
                source=source, url=f"https://{source}.acme.local/{doc_id}", acl=acl[a],
                updated_at=updated, authority=authority, status="active", superseded_by=None)

            def swapped(pid, text, suffix):
                q, year, wrong = changed[a]
                return {"id": f"{pid}-{suffix}",
                        "text": year_pattern(year).sub(wrong, text) if pid == q["pid"] else text}

            if a in superseded:
                old_id = f"{doc_id}-old"
                add(old_id, art["title"], lang, [swapped(p["id"], p["text"], "old") for p in paras],
                    source=source, url=f"https://{source}.acme.local/{old_id}", acl=acl[a],
                    updated_at=updated - timedelta(days=rng.randint(400, 1100)), authority=authority,
                    status="archived", superseded_by=doc_id)
            if a in informal:
                pid = changed[a][0]["pid"]
                slack_id = f"{doc_id}-slack"
                text = next(p["text"] for p in paras if p["id"] == pid)
                add(slack_id, f"{art['title']} (Slack)", lang, [swapped(pid, text, "slack")],
                    source="slack", url=f"https://slack.acme.local/{slack_id}", acl=acl[a],
                    updated_at=updated + timedelta(days=rng.randint(30, 90)), authority="informal",
                    status="active", superseded_by=None)
            if a in duplicated:
                dup_id = f"{doc_id}-dup"
                add(dup_id, art["title"], lang, [{"id": f"{p['id']}-dup", "text": p["text"]} for p in paras],
                    source="gdrive", url=f"https://gdrive.acme.local/{dup_id}", acl=acl[a],
                    updated_at=random_date(rng, date(2022, 1, 1), date(2026, 6, 30)), authority="verified",
                    status="active", superseded_by=None)

    generation_qids = {q["qid"] for q in rng.sample(
        [q for q in questions if q["art"] in open_arts and q["art"] not in used], N_GENERATION)}
    restricted = [q for q in questions if acl[q["art"]] != ["all-employees"]]
    acl_qs = rng.sample(restricted, N_ACL)

    golden = []
    for q in questions:
        tags, extra = [], {}
        if q["qid"] in generation_qids:
            tags.append("generation")
        if q["art"] in duplicated:
            tags.append("duplicated")
        if q["art"] in changed and changed[q["art"]][0]["qid"] == q["qid"]:
            tags.append("superseded" if q["art"] in superseded else "informal")
            extra["wrong_answer"] = changed[q["art"]][2]
        for lang in LANGS:
            golden.append({"id": f"{q['qid']}-{lang}", "kind": "answerable", "lang": lang, "user": "root",
                           "question": q[lang]["question"], "answers": q[lang]["answers"],
                           "gold_paragraph": q["pid"], "gold_doc": f"{q['art']}-{lang}",
                           "tags": tags, **extra})
    for q in acl_qs:
        for lang in LANGS:
            golden.append({"id": f"{q['qid']}-{lang}-acl", "kind": "acl", "lang": lang, "user": "alice",
                           "question": q[lang]["question"], "answers": q[lang]["answers"],
                           "gold_paragraph": q["pid"], "gold_doc": f"{q['art']}-{lang}", "tags": []})
    for i, pair in enumerate(UNANSWERABLE):
        for lang, question in zip(LANGS, pair):
            golden.append({"id": f"unans{i:02d}-{lang}", "kind": "unanswerable", "lang": lang, "user": "root",
                           "question": question, "answers": [], "gold_paragraph": None, "gold_doc": None,
                           "tags": []})
    return docs, meta, golden


def main():
    docs, meta, golden = build(random.Random(SEED))
    corpus = DATA / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for old in corpus.glob("*.json"):
        old.unlink()
    for d in docs:
        (corpus / f"{d['doc_id']}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    (corpus / "meta.yaml").write_text(yaml.dump(meta, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True))
    (DATA / "users.yaml").write_text(yaml.safe_dump(USERS, sort_keys=False))
    with (DATA / "golden.jsonl").open("w") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{len(docs)} docs, {len(golden)} golden questions -> {DATA}")


if __name__ == "__main__":
    main()
