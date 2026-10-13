# Implementation Plan — Toy RAG Q&A Bot

Goal: a working, local, small-scale version of the system in [design.md](design.md). The scale is toy-sized, but the serving pipeline keeps all its parts: hybrid retrieval, ACL (Access Control List) filtering, a cross-encoder reranker, post-ranking, a confidence gate, grounded generation with citations, multi-turn conversations and offline metrics. It works in **English and Russian**, including cross-lingual questions such as a Russian question answered from an English chunk.

Target machine: RTX 4090 (24 GB VRAM), 64 GB RAM, Docker, Python ≥ 3.10, uv.

## 1. Scope

**In scope** (from design.md → "Serving pipeline" and "Indexing pipeline"):

| Design component | Toy implementation |
|---|---|
| Outer sources + Document Update Service | XQuAD articles exported to JSON/markdown files with metadata in `data/corpus/`; sync by content hash (new / changed / deleted / ACL-only change) |
| Preprocess + chunking | Paragraph-aware chunker: a chunk is never cut mid-paragraph unless a paragraph is longer than the limit. ~300 tokens, ~15% overlap, token counts from the embedding model's tokenizer. Each chunk stores the `paragraph_ids` it contains, which link it to the relevance labels |
| Chunk/Query Encoder (bi-encoder) | `BAAI/bge-m3` (multilingual, 1024-dim; the design assumed 768), via sentence-transformers. One shared vector space for EN and RU |
| Vector index | Qdrant (Docker), HNSW (Hierarchical Navigable Small World), cosine |
| BM25 index | Qdrant sparse vectors with the IDF (Inverse Document Frequency) modifier, stored in the same collection, so there is a single database. Tokenization is **language-aware**: the language of each chunk and query is detected, then an English or Russian Snowball stemmer is applied (fastembed `Qdrant/bm25` with `language=`) |
| ACL filter inside search | Qdrant payload filter `acl ∈ user_groups` applied in **both** the dense and sparse searches; a chunk with no ACL never matches (fail closed) |
| RRF (Reciprocal Rank Fusion) | Implemented in Python (`Σ w_l / (k + rank)`, k=60) so the mechanism stays visible, not hidden inside Qdrant's built-in fusion |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` (multilingual), top 50 → top 20. Smaller option closer to the design's MiniLM: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Post-ranking | freshness (per-source half-life), authority prior, lifecycle (archived/superseded → excluded), per-doc cap, MMR (Maximal Marginal Relevance) dedup → top 5–8 |
| Confidence gate | Threshold on the **raw** cross-encoder score → "not found" without calling the LLM (Large Language Model). The threshold is calibrated on the eval set |
| Query condensation | An LLM call rewrites (history + follow-up) into a standalone query, keeping the user's language; if it fails, the raw query is used |
| Prompt generation | System rules + numbered chunks with source tags; the strongest chunk is placed first; "answer in the language of the question" |
| LLM | vLLM (Docker, `vllm/vllm-openai`) serving `Qwen/Qwen2.5-7B-Instruct-AWQ` (good Russian; about 5.5 GB of weights, leaving VRAM for the KV cache, embedder and reranker). Called through the OpenAI-compatible API with streaming. The model name is configurable |
| API | FastAPI: `POST /v1/ask` streams over SSE (Server-Sent Events): tokens, then a final `citations` event; `conversation_id` for multi-turn; `POST /v1/admin/sync`; `GET /health` |
| Auth | Toy auth: an `X-User` header mapped to groups via `data/users.yaml`; an unknown user or missing header → 401 (fail closed) |
| Citations | Parse `[n]` markers → map them to chunk title/url/snippet; drop invalid markers |
| Fallbacks | LLM down → return ranked links; one retrieval branch down → use the other; auth unknown → refuse |
| Offline metrics | Recall@k, MRR (Mean Reciprocal Rank), NDCG@k (Normalized Discounted Cumulative Gain) per stage; optional LLM-judge faithfulness |

**Out of scope:** real connectors, OCR (Optical Character Recognition), A/B tests, monitoring dashboards, a UI. A CLI (Command-Line Interface) is kept only for ops commands: `ingest`, `eval`, `serve`.

## 2. Corpus — XQuAD EN + RU

Sources and licenses are in [datasets.md](datasets.md). A `scripts/prepare_corpus.py` script downloads the data from Hugging Face and builds:

- **Documents.** XQuAD has 240 paragraphs from 48 Wikipedia articles. Joining with SQuAD v1.1 dev on the question `id` recovers each article's title. One document = one article in one language, so 48 × 2 = 96 docs. Paragraph IDs are stable, e.g. `art07-p03`, and are shared between the EN and RU versions.
- **Synthetic metadata,** generated deterministically with a fixed seed into `data/corpus/meta.yaml`. Real Wikipedia has none of this, and every post-ranking and permission stage needs something to act on:

  ```yaml
  doc_id: art07-en
  title: Super Bowl 50
  lang: en
  source: confluence          # confluence | slack | gdrive | wiki | ticket
  url: https://wiki.acme.local/art07-en
  acl: [all-employees]        # some articles get [research] or [hr-team]
  updated_at: 2025-03-01
  authority: official         # official | verified | informal
  status: active              # active | archived | deprecated
  superseded_by: null
  ```

- **Edge cases built into the corpus on purpose:**
  - **Restricted** articles (`acl: [hr-team]`) → permission-leak tests.
  - **Superseded copies:** for a few articles, an older version with one fact changed (e.g. a different year), marked `archived` with `superseded_by` set. The answer must come from the active version.
  - **Duplicates:** an article copied into a second "space" under a different `doc_id` → dedup and the per-doc cap.
  - **Informal copies:** a `slack` copy of a paragraph with a contradicting fact and `authority: informal` → authority prior.
- **Users** — `data/users.yaml`: e.g. `alice: [all-employees]`, `bob: [all-employees, research]`, `hana: [all-employees, hr-team]`.
- **Golden set** — `data/golden.jsonl`:
  - **Retrieval eval:** all 1,190 questions × 2 languages. Relevance is graded: the gold paragraph in the **same** language = 2, its parallel paragraph in the other language = 1. NDCG uses the grades; a chunk counts as a hit if its `paragraph_ids` contain the gold paragraph.
  - **Generation eval:** about 60 sampled questions (EN + RU) with the XQuAD answer span as the reference answer.
  - **Unanswerable:** about 30 hand-written questions (EN + RU) that the corpus doesn't cover → "not found" expected.
  - **ACL cases:** questions whose gold doc is restricted, asked as `alice` → "not found" expected, and the restricted doc never appears in sources.
  - **Cross-lingual split:** RU questions against an EN-only index, and the reverse, to measure what the dense branch carries on its own. BM25 can't match across scripts, which is a useful thing to see in the per-branch metrics.

## 3. Project layout

```
QA/
├── pyproject.toml            # uv-managed
├── docker-compose.yml        # qdrant + vllm (GPU) + api
├── config.yaml               # models, k values, weights, thresholds
├── scripts/prepare_corpus.py # download XQuAD/SQuAD → data/
├── data/{corpus/, users.yaml, golden.jsonl}
├── src/rag/
│   ├── config.py             # pydantic settings loaded from config.yaml
│   ├── lang.py               # language detection (Cyrillic ratio heuristic)
│   ├── cli.py                # `rag ingest`, `rag eval`, `rag serve`
│   ├── storage/              # the Qdrant index and what writes vectors into it
│   │   ├── encoders.py       # dense (bge-m3) + sparse (BM25, per-language stemmer)
│   │   └── index.py          # Qdrant: create collection, upsert, delete, set_payload
│   ├── indexing/             # corpus → index (offline)
│   │   ├── documents.py      # load corpus + metadata → Document
│   │   ├── chunking.py       # paragraph-aware chunker
│   │   └── ingest.py         # Document Update Service: diff by hash, sync index
│   ├── search/               # query → ranked chunks
│   │   ├── retrieval.py      # dense search, sparse search, RRF
│   │   ├── rerank.py         # cross-encoder + confidence gate
│   │   └── postrank.py       # freshness / authority / lifecycle / dedup / cap
│   ├── generation/           # chunks → answer
│   │   ├── llm.py            # OpenAI-compatible streaming client (vLLM)
│   │   ├── query.py          # multi-turn condensation
│   │   ├── prompt.py         # prompt assembly + token budget
│   │   └── citations.py      # [n] → sources
│   ├── serving/              # HTTP + orchestration
│   │   ├── pipeline.py       # serving orchestration + fallbacks
│   │   ├── api.py            # FastAPI app (SSE streaming)
│   │   ├── auth.py           # user → groups, fail closed
│   │   └── conversations.py  # in-memory conversation store
│   ├── eval/                 # Phase 8: Recall@k, MRR, NDCG, (judge)
│   └── train/                # Phase 9: InfoNCE fine-tuning
└── tests/{unit/, integration/, e2e/}
```

The interfaces are small and swappable (`Encoder`, `Reranker`, `LLM` protocols), so unit tests can use fakes instead of real models.

## 4. Phases

Each phase ends with a check that can be run.

1. **Skeleton + infra** — uv project, `docker-compose.yml` (Qdrant + vLLM with GPU), config, pytest setup.
   → verify: `docker compose up`, Qdrant answers on `:6333`, `curl :8000/v1/chat/completions` returns a completion in both EN and RU.
2. **Corpus prep + loader + chunker** — `prepare_corpus.py` (XQuAD + SQuAD title join + synthetic metadata + edge cases + golden set), `documents.py`, `chunking.py`, `lang.py`.
   → verify: unit tests: 96 base docs, 240 paragraph IDs per language, EN/RU paragraph IDs match one to one; chunk size ≤ limit, overlap present, every paragraph is covered by ≥ 1 chunk, ACL inherited by chunks; language detection on EN/RU samples.
3. **Encoders + indexing + Document Update Service** — dense and sparse encoders; a Qdrant collection with named vectors `dense` + `sparse`; hash-based sync.
   → verify: integration tests (real Qdrant): ingest → N points; edit a doc → only its chunks are re-embedded; ACL-only change → payload update without re-embedding; delete a doc → its chunks are gone; a Russian word-form variant ("договора" vs "договор") matches through BM25 stemming.
4. **Hybrid retrieval + ACL + RRF** — both searches filtered by ACL, RRF fusion in Python.
   → verify: unit tests for RRF math (appearing in both lists beats rank 1 in one); integration **permission-leak canary**: `alice` never gets `hr-team` chunks from either branch; missing ACL → excluded; a cross-lingual query is found by dense only.
5. **Reranker + post-ranking + confidence gate** — cross-encoder, then the blend `s_ce_norm + w_f·freshness + w_a·authority − w_d·dup`, lifecycle exclusion, per-doc cap, gate on the raw CE (cross-encoder) score.
   → verify: unit tests: superseded doc excluded; the informal contradicting copy ranks below the official one; freshness only reorders near-ties and never lifts an off-topic chunk; cap respected; gate fires on unanswerable questions.
6. **Generation** — prompt assembly with a token budget, streaming vLLM client, citation mapping, condensation for follow-ups.
   → verify: unit tests with a fake LLM: prompt contains numbered chunks, invalid `[7]` markers are dropped, the condensation fallback works; e2e (real vLLM): an answer contains ≥ 1 valid citation; an RU question gets an RU answer; an unanswerable question → "not found".
7. **Pipeline + FastAPI + fallbacks** — wire up the 13 serving steps behind `/v1/ask` with SSE streaming; conversation store.
   → verify: API tests with `TestClient` + fakes: SSE event order (tokens → citations → done), 401 without `X-User`, a follow-up in the same `conversation_id` gets condensed; e2e: LLM down → links returned; dense branch disabled → still answers from sparse.
8. **Offline evaluation** — `rag eval` prints Recall@{5,10}, MRR and NDCG@10 **per stage** (dense only, sparse only, after RRF, after rerank, after post-rank) and **per language pair** (EN→EN, RU→RU, RU→EN, EN→RU); optional LLM-judge faithfulness on the generation subset.
   → verify: unit tests of the metric functions against hand-computed values; a regression test asserts Recall@5 ≥ a threshold.
9. **Train the bi-encoder (InfoNCE)** — implement the symmetric InfoNCE loss from design.md by hand: in-batch negatives, a learnable log-temperature that is clamped, plus hard negatives mined from the current model's ANN (Approximate Nearest Neighbor) top-k.
   - Training data: SQuAD v1.1 **train** (EN) + SberQuAD train (RU) as (question, paragraph) pairs; MIRACL `en`/`ru` judged non-relevant passages as extra hard negatives. Never the SQuAD dev split, which XQuAD comes from.
   - Model: fine-tune `bge-m3` (568M params; bf16 + gradient checkpointing fits on a 4090), or `intfloat/multilingual-e5-small` for fast iterations.
   - Re-index with the new model. The embedding model version is stored in the collection metadata, and the API refuses to start on a mismatch (the design's "deploy block").
   → verify: unit test of the loss against a hand-computed small batch (symmetry, positive included in the denominator); `rag eval` compares Recall@k/NDCG before and after. bge-m3 is already strong on Wikipedia, so gains may be small or negative; the before/after comparison is the point.

## 5. Test strategy

- **unit/** — no network, no models (fakes); fast, run on every change.
- **integration/** — real Qdrant (the Docker container, or `QdrantClient(":memory:")` where its features are enough) plus real encoders on a small corpus slice; marked `@pytest.mark.integration`.
- **e2e/** — the full pipeline with vLLM; marked `@pytest.mark.llm` and skipped automatically when vLLM isn't reachable. Assertions check structure (citations valid, answer language, "not found" on unanswerable questions, no leaks), not exact wording, because LLM output varies.
- The security tests (ACL leak canary, fail-closed auth) are always run, never skipped.

## 6. Stack summary

Python 3.10+, uv, qdrant-client, sentence-transformers (`bge-m3`, `bge-reranker-v2-m3`), fastembed (BM25 sparse with EN/RU stemmers), vLLM (`Qwen2.5-7B-Instruct-AWQ`), openai (client library only, pointed at vLLM), FastAPI + uvicorn (SSE), pydantic, typer (ops CLI), datasets (Hugging Face), PyTorch (Phase 9), pytest + httpx.
