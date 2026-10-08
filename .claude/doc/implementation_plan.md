# Implementation Plan — Toy RAG Q&A Bot

Goal: a working, local, small-scale version of the system in [design.md](design.md). The scale is toy-sized, but the serving pipeline keeps all its parts: hybrid retrieval, ACL (Access Control List) filtering, a cross-encoder reranker, post-ranking, a confidence gate, grounded generation with citations, multi-turn conversations and offline metrics.

Target machine: RTX 4090 (24 GB VRAM), 64 GB RAM, Docker, Python ≥ 3.10, uv.

## 1. Scope

**In scope** (from design.md → "Serving pipeline" and "Indexing pipeline"):

| Design component | Toy implementation |
|---|---|
| Outer sources + Document Update Service | Markdown files with YAML frontmatter in `data/corpus/`; sync by content hash (new / changed / deleted / ACL-only change) |
| Preprocess + chunking | Heading-aware chunker, ~300 tokens, ~15% overlap, token counts from the embedding model's tokenizer |
| Chunk/Query Encoder (bi-encoder) | `BAAI/bge-base-en-v1.5` (768-dim, the same size as in the design), via sentence-transformers |
| Vector index | Qdrant (Docker), HNSW (Hierarchical Navigable Small World), cosine |
| BM25 index | Qdrant sparse vectors with the IDF (Inverse Document Frequency) modifier (fastembed `Qdrant/bm25`), stored in the same collection, so there is a single database |
| ACL filter inside search | Qdrant payload filter `acl ∈ user_groups` applied in **both** the dense and sparse searches; a chunk with no ACL never matches (fail closed) |
| RRF (Reciprocal Rank Fusion) | Implemented in Python (`Σ w_l / (k + rank)`, k=60) so the mechanism stays visible, not hidden inside Qdrant's built-in fusion |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 50 → top 20 |
| Post-ranking | freshness (per-source half-life), authority prior, lifecycle (archived/superseded → excluded), per-doc cap, MMR (Maximal Marginal Relevance) dedup → top 5–8 |
| Confidence gate | Threshold on the **raw** cross-encoder score → "not found" without calling the LLM (Large Language Model) |
| Query condensation | An LLM call rewrites (history + follow-up) into a standalone query; if it fails, the raw query is used |
| Prompt generation | System rules + numbered chunks with source tags; the strongest chunk is placed first |
| LLM | Local model served by Ollama (Docker, GPU) through its OpenAI-compatible API with streaming. Default `qwen2.5:7b-instruct` (also fits: `qwen3:8b`, or a 14B model in Q4) |
| Citations | Parse `[n]` markers → map them to chunk title/url/snippet; drop invalid markers |
| Fallbacks | LLM down → return ranked links; one retrieval branch down → use the other; auth unknown → refuse |
| Offline metrics | Recall@k, MRR (Mean Reciprocal Rank), NDCG@k (Normalized Discounted Cumulative Gain) on a golden set; optional LLM-judge faithfulness |

**Out of scope** (or optional later phases): real connectors, OCR (Optical Character Recognition), training the bi-encoder/cross-encoder (see Phase 9, optional), A/B tests, monitoring dashboards, a UI.

## 2. Toy corpus

A fictional company, "Acme", with ~25 short markdown docs spread over the design's source types, each with frontmatter:

```yaml
doc_id: hr-vacation-2024
title: Vacation Policy (2024)
source: confluence          # confluence | slack | gdrive | wiki | ticket
url: https://wiki.acme.local/hr/vacation
acl: [all-employees]        # groups allowed to see it
updated_at: 2024-03-01
authority: official         # official | verified | informal
status: active              # active | archived | deprecated
superseded_by: null
```

Edge cases built into the corpus on purpose, so every pipeline stage has something to test:
- an **old vacation policy** superseded by a new one → lifecycle exclusion
- a **restricted** doc (`acl: [hr-team]`, e.g. salary bands) → permission-leak tests
- a Slack message that contradicts an official doc → authority prior
- a doc with error codes / ticket IDs (`ERR-4031`, `OPS-1234`) → where the lexical branch wins
- paraphrase-only matches ("get reimbursed" ↔ "Expense policy") → where the dense branch wins
- the same text copy-pasted into two spaces → dedup
- a long doc → multiple chunks, per-doc cap

Plus:
- `data/users.yaml` — toy users → groups (e.g. `alice: [all-employees, engineering]`, `hana: [all-employees, hr-team]`)
- `data/golden.jsonl` — ~30 `(question, relevant_doc_ids, reference_answer, user)` rows, including unanswerable questions and questions that should be blocked by ACL

Relevance is labeled at the **doc** level rather than the chunk level, so the labels don't break when chunking parameters change. Retrieval metrics map each chunk to its parent doc.

## 3. Project layout

```
QA/
├── pyproject.toml            # uv-managed
├── docker-compose.yml        # qdrant + ollama (GPU)
├── config.yaml               # models, k values, weights, thresholds
├── data/{corpus/, users.yaml, golden.jsonl}
├── src/rag/
│   ├── config.py             # pydantic settings loaded from config.yaml
│   ├── documents.py          # load md + frontmatter → Document
│   ├── chunking.py           # heading-aware chunker
│   ├── encoders.py           # dense (bge) + sparse (bm25) encoders
│   ├── index.py              # Qdrant: create collection, upsert, delete, set_payload
│   ├── ingest.py             # Document Update Service: diff by hash, sync index
│   ├── auth.py               # user → groups, fail closed
│   ├── retrieval.py          # dense search, sparse search, RRF
│   ├── rerank.py             # cross-encoder
│   ├── postrank.py           # freshness / authority / lifecycle / dedup / cap
│   ├── query.py              # multi-turn condensation
│   ├── prompt.py             # prompt assembly + token budget
│   ├── llm.py                # OpenAI-compatible streaming client
│   ├── citations.py          # [n] → sources
│   ├── pipeline.py           # serving orchestration + fallbacks
│   ├── eval.py               # Recall@k, MRR, NDCG, (judge)
│   └── cli.py                # `rag ingest`, `rag ask`, `rag chat`, `rag eval`
└── tests/{unit/, integration/, e2e/}
```

The interfaces are small and swappable (`Encoder`, `Reranker`, `LLM` protocols), so unit tests can use fakes instead of real models.

## 4. Phases

Each phase ends with a check that can be run.

1. **Skeleton + infra** — uv project, `docker-compose.yml` (Qdrant + Ollama with GPU), config, pytest setup.
   → verify: `docker compose up`, Qdrant answers on `:6333`, `ollama run qwen2.5:7b-instruct "hi"` works.
2. **Corpus + loader + chunker** — write the toy corpus, users and golden set; implement `documents.py` and `chunking.py`.
   → verify: unit tests for frontmatter parsing, chunk size ≤ limit, overlap present, no chunk crosses a heading boundary, ACL inherited by chunks.
3. **Encoders + indexing + Document Update Service** — dense and sparse encoders; a Qdrant collection with named vectors `dense` + `sparse`; hash-based sync.
   → verify: integration tests (real Qdrant): ingest → N points; edit a doc → only its chunks are re-embedded; ACL-only change → payload update without re-embedding; delete a doc → its chunks are gone.
4. **Hybrid retrieval + ACL + RRF** — both searches filtered by ACL, RRF fusion in Python.
   → verify: unit tests for RRF math (appearing in both lists beats rank 1 in one); integration **permission-leak canary**: a low-permission user never gets `hr-team` chunks; missing ACL → excluded; error-code query found by sparse, paraphrase query found by dense.
5. **Reranker + post-ranking + confidence gate** — cross-encoder, then the blend `s_ce_norm + w_f·freshness + w_a·authority − w_d·dup`, lifecycle exclusion, per-doc cap, gate on the raw CE (cross-encoder) score.
   → verify: unit tests: superseded doc excluded; freshness only reorders near-ties and never lifts an off-topic chunk; cap respected; gate fires on an off-topic question.
6. **Generation** — prompt assembly with a token budget, streaming LLM client, citation mapping, condensation for follow-ups.
   → verify: unit tests with a fake LLM: prompt contains numbered chunks, invalid `[7]` markers are dropped, the condensation fallback works; e2e (real Ollama): an answer contains ≥ 1 valid citation; an unanswerable question → "not found".
7. **Pipeline + CLI + fallbacks** — wire up the 13 serving steps; `rag ask --user alice "…"`, `rag chat`.
   → verify: e2e tests: LLM down → links returned; dense branch disabled → still answers from sparse; unknown user → refused.
8. **Offline evaluation** — `rag eval` prints Recall@{5,10}, MRR and NDCG@10 per stage (after RRF, after rerank, after post-rank), so you can see what each stage contributes; optional LLM-judge faithfulness.
   → verify: unit tests of the metric functions against hand-computed values; a regression test asserts Recall@5 ≥ a threshold on the golden set.
9. *(Optional, for study)* **Train the bi-encoder** — generate synthetic queries per chunk with the LLM (doc2query), fine-tune bge with symmetric InfoNCE plus hard negatives mined from ANN (Approximate Nearest Neighbor) top-k, then compare Recall@k before and after. With a toy corpus this teaches the mechanics; don't expect large gains.

## 5. Test strategy

- **unit/** — no network, no models (fakes); fast, run on every change.
- **integration/** — real Qdrant (the Docker container, or `QdrantClient(":memory:")` where its features are enough) plus real small encoders; marked `@pytest.mark.integration`.
- **e2e/** — the full pipeline with Ollama; marked `@pytest.mark.llm` and skipped automatically when Ollama isn't reachable. Assertions check structure (citations valid, "not found" on unanswerable questions, no leaks), not exact wording, because LLM output varies.
- The security tests (ACL leak canary, fail-closed) are always run, never skipped.

## 6. Stack summary

Python 3.10+, uv, qdrant-client, sentence-transformers (bge-base, MiniLM cross-encoder), fastembed (BM25 sparse), Ollama (`qwen2.5:7b-instruct`), openai (client library only, pointed at Ollama), pydantic, typer (CLI), pytest.
