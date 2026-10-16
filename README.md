# Internal Knowledge Q&A Bot (RAG, EN + RU)

A toy but complete RAG (Retrieval-Augmented Generation) bot that answers employees' questions from a permission-gated document corpus. Every answer is grounded in retrieved passages, carries clickable citations, and is never built from documents the asking user can't see. English and Russian are both supported.

The corpus is XQuAD (SQuAD v1.1 dev articles, parallel EN/RU) dressed up with synthetic company metadata: source system, ACL (Access Control List), dates, authority, lifecycle flags. See [scripts/prepare_corpus.py](scripts/prepare_corpus.py).

Design rationale lives in [.claude/doc/design.md](.claude/doc/design.md); known limitations in [.claude/doc/todo.md](.claude/doc/todo.md).

## What it guarantees

- **Grounded + cited**: the LLM (Large Language Model) may use only the numbered passages, and every `[n]` marker is mapped back to a real chunk, URL and snippet. Invalid markers are stripped.
- **"Not found" instead of hallucination**: a confidence gate on the cross-encoder score and a `NOT_FOUND` sentinel in the prompt.
- **ACL enforced inside the search**: the permission filter is part of both Qdrant queries, not applied afterwards. A chunk with a missing or empty ACL is never visible (fail closed). Unknown users get 401.
- **Multi-turn**: follow-ups are rewritten into standalone questions before retrieval; history is keyed per user.
- **Graceful degradation**: one search branch down → the other serves alone. LLM down → ranked source links are returned. Condensation fails → the raw question is used.
- **Freshness**: sync re-embeds only documents whose content changed; metadata-only changes (e.g. ACL) are a cheap payload update.

## How a question is answered

```
                 POST /v1/ask   (header X-User: alice)
                          │
                          ▼
   ┌─────────────────────────────────────────────┐
   │ 1. auth          X-User → groups (401 if    │  serving/auth.py
   │                  unknown)                   │
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 2. condense      history + follow-up →      │  generation/query.py
   │                  standalone query (LLM)     │  serving/conversations.py
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 3. retrieve      ┌─ dense  (bge-m3) ──┐     │  search/retrieval.py
   │    ACL filter    │                    ├─RRF─┼─► top 50 candidates
   │    in both       └─ sparse (BM25)  ───┘     │  storage/encoders.py
   │    branches                                 │  storage/index.py
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 4. rerank        cross-encoder              │  search/rerank.py
   │                  bge-reranker-v2-m3 → 20    │
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 5. post-rank     drop archived/superseded;  │  search/postrank.py
   │                  + freshness + authority;   │
   │                  MMR dedup; ≤2 per doc → 8  │
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 6. gate          best raw CE logit          │  search/rerank.py
   │                  < -2.0 ?  ──yes──► "not_found" (no LLM call)
   └─────────────────────┬───────────────────────┘
                         ▼ no
   ┌─────────────────────────────────────────────┐
   │ 7. prompt        system rules + numbered    │  generation/prompt.py
   │                  chunks + history + question│  (token budget)
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 8. generate      vLLM (Qwen2.5-7B-AWQ),     │  generation/llm.py
   │                  streamed; NOT_FOUND held   │  serving/pipeline.py
   │                  back from the stream       │
   └─────────────────────┬───────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────┐
   │ 9. citations     [n] → chunk/url/snippet    │  generation/citations.py
   └─────────────────────┬───────────────────────┘
                         ▼
        SSE events:  meta → token* → result → done
```

Stage details:

| # | Stage | What happens |
|---|-------|--------------|
| 2 | Condense | With no history the question passes through unchanged. Otherwise the last `history_turns` messages + the question are rewritten by the LLM into a standalone query. Failure falls back to the raw question. |
| 3 | Retrieve | Dense (bge-m3 cosine) and sparse (BM25 with per-language stemmer, IDF applied by Qdrant) each return `candidates` (50) chunks, both filtered by the user's groups. Fused by RRF (Reciprocal Rank Fusion): `score = Σ w / (k + rank)`, `k = 60`. |
| 4 | Rerank | A cross-encoder scores `(query, chunk)` pairs jointly and outputs a raw logit. Top 20 are kept. |
| 5 | Post-rank | `final = minmax(ce) + 0.05·freshness + 0.1·authority − 0.3·similarity_to_picked`. Archived or superseded docs are excluded outright. Near-duplicates (cosine ≥ 0.95) are dropped; max 2 chunks per document; 8 chunks kept. Freshness uses a per-source half-life (Slack 30 days … Confluence 730 days). |
| 6 | Gate | Uses the **raw** cross-encoder score (not the post-ranked one): a boosted chunk is not evidence that the question is answerable. Threshold `-2.0` was calibrated on the golden set. |
| 7 | Prompt | Chunks are added in rank order until the token budget (`context_window − max_tokens`) is spent, then the most recent history turns. System rules and question always fit. |
| 8 | Generate | Streams tokens. The start of the answer is held back until it's clear it isn't `NOT_FOUND`. After streaming, the full answer is checked: `NOT_FOUND` anywhere → `not_found`; only `[n]` markers → ranked links; wrong language (Cyrillic in an English answer, or an English sentence in a Russian one) → one retry with a language reminder, then ranked links. Because a checked answer may differ from what was streamed, clients should display the `result` event, not the concatenated tokens. |
| 9 | Citations | `[n]` → chunk metadata, listed once each in order of first appearance. |

### Result statuses

| `status` | Meaning |
|----------|---------|
| `answered` | LLM answered; `citations` lists the sources |
| `not_found` | Gate blocked the question, or the LLM replied `NOT_FOUND` |
| `llm_unavailable` | LLM call failed, or its answer was unusable (no text, or still in the wrong language after one retry); `citations` holds the ranked links instead |

## How the index is built

```
 data/corpus/*.json  ──┐   (content: paragraphs)
                       ├─► load_corpus ─► Document ─► chunk ─► embed ─► Qdrant
 data/corpus/meta.yaml ┘   (ACL, source,   (indexing/  (indexing/  (storage/
                            dates, status)  documents)  chunking)   encoders)
```

[`rag ingest`](src/rag/cli.py) runs `DocumentUpdateService.sync()`, which diffs the corpus against the index by hash:

| Situation | Action |
|-----------|--------|
| Doc not in the index | chunk + embed (`added`) |
| Content/embedder/chunking hash changed | re-embed that doc only (`changed`) |
| Only metadata hash changed (ACL, dates, status…) | payload update, no re-embedding (`meta_only`) |
| Doc no longer in the corpus | its chunks are deleted (`deleted`) |
| Nothing changed | skipped (`unchanged`) |

Chunking: one paragraph = one chunk; a paragraph over 300 tokens is split into sentence windows with ~15% overlap. Each Qdrant point holds a `dense` vector (cosine), a `sparse` BM25 vector, and a payload with chunk text, doc metadata, ACL and hashes. The collection records which embedder built it; the app refuses to start against vectors from a different model (re-index with `rag ingest --reset`).

## Project layout

```
config.yaml                  every tunable (models, chunking, retrieval, rerank, postrank, train)
docker-compose.yml           Qdrant + vLLM
run.sh                       start everything: compose, API and GUI
scripts/prepare_corpus.py    build data/ from XQuAD (deterministic, seed 42)
data/
  corpus/<doc_id>.json       document content (48 articles × EN/RU, plus old/dup/slack variants)
  corpus/meta.yaml           per-doc metadata: ACL, source, updated_at, authority, status
  users.yaml                 user → groups (alice, bob, hana, root, ops)
  golden.jsonl               2430 eval questions: answerable / unanswerable / acl
  eval/baseline.json         saved offline metrics of the base bge-m3
models/                      fine-tuned embedder (git-ignored)
tests/                       unit / integration / e2e
```

Source modules ([src/rag/](src/rag/)):

| Stage | File | Role |
|-------|------|------|
| Entry | [cli.py](src/rag/cli.py) | `rag` commands: `ingest`, `serve`, `eval`, `eval-generation`, `train` |
| | [config.py](src/rag/config.py) | Typed loader for `config.yaml` |
| | [lang.py](src/rag/lang.py) | EN/RU detection (≥50% Cyrillic letters → `ru`) |
| Indexing | [indexing/documents.py](src/rag/indexing/documents.py) | `Document` model; joins content JSON with `meta.yaml` |
| | [indexing/chunking.py](src/rag/indexing/chunking.py) | Paragraph → chunk(s) with sentence windows and overlap |
| | [indexing/ingest.py](src/rag/indexing/ingest.py) | Hash-based incremental sync (`DocumentUpdateService`) |
| | [indexing/importer.py](src/rag/indexing/importer.py) | Admin web-page import: download (Wikipedia API / trafilatura), clean, dedupe, write corpus files, sync |
| Storage | [storage/encoders.py](src/rag/storage/encoders.py) | `BgeM3Encoder` (dense), `Bm25Encoder` (sparse) |
| | [storage/index.py](src/rag/storage/index.py) | Qdrant wrapper: collection, upsert, delete, payload update, embedder guard |
| Search | [search/retrieval.py](src/rag/search/retrieval.py) | ACL-filtered dense + sparse search, RRF fusion, branch fallback |
| | [search/rerank.py](src/rag/search/rerank.py) | Cross-encoder reranker and confidence gate |
| | [search/postrank.py](src/rag/search/postrank.py) | Freshness / authority / lifecycle / MMR dedup / per-doc cap |
| Generation | [generation/query.py](src/rag/generation/query.py) | Multi-turn condensation |
| | [generation/prompt.py](src/rag/generation/prompt.py) | System prompt, token-budgeted context assembly |
| | [generation/llm.py](src/rag/generation/llm.py) | Streaming client for vLLM's OpenAI-compatible API |
| | [generation/citations.py](src/rag/generation/citations.py) | `[n]` → source mapping |
| Serving | [serving/pipeline.py](src/rag/serving/pipeline.py) | Orchestrates stages 2–9, emits events, fallbacks |
| | [serving/api.py](src/rag/serving/api.py) | FastAPI: `/v1/ask` (SSE), `/v1/me`, `/v1/admin/sync`, `/v1/admin/import`, `/health` |
| | [serving/auth.py](src/rag/serving/auth.py) | `X-User` → groups, fail closed |
| | [serving/conversations.py](src/rag/serving/conversations.py) | In-memory history keyed by (user, conversation) |
| UI | [ui/app.py](src/rag/ui/app.py) | Streamlit GUI: login, chat with citations, admin sync and web-page import |
| Eval | [eval/runner.py](src/rag/eval/runner.py) | Per-stage Recall@5/10, MRR, NDCG@10 and gate calibration |
| | [eval/metrics.py](src/rag/eval/metrics.py) | Metric functions over graded relevance |
| | [eval/generation.py](src/rag/eval/generation.py) | Dumps pipeline answers + exact prompts for an LLM judge |
| Training | [train/data.py](src/rag/train/data.py) | SQuAD / SberQuAD / MIRACL triplets with mined hard negatives |
| | [train/loss.py](src/rag/train/loss.py) | Symmetric InfoNCE with learnable temperature |
| | [train/train.py](src/rag/train/train.py) | Fine-tuning loop (bf16, gradient checkpointing) |

## Install

**Requirements**: Linux, Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), Docker with the NVIDIA runtime, and one GPU. The compose file budgets 24 GB: ~60% for vLLM, the rest for the embedder and reranker on the host. Training the embedder also needs a CUDA GPU.

```bash
# 1. Python dependencies
uv sync

# 2. Qdrant + vLLM (first start downloads Qwen2.5-7B-Instruct-AWQ)
export HF_TOKEN=...            # optional, avoids Hugging Face rate limits
docker compose up -d
curl localhost:6333/readyz     # Qdrant
curl localhost:8000/v1/models  # vLLM

# 3. Embedder choice (models/ is git-ignored, so a fresh clone doesn't have the fine-tuned one):
#    either set `models.embedder: BAAI/bge-m3` in config.yaml,
#    or fine-tune it yourself:  uv run rag train

# 4. Index the corpus
uv run rag ingest --reset
```

To regenerate `data/` from scratch: `uv run python scripts/prepare_corpus.py` (downloads XQuAD and SQuAD titles).

## Run

### One command (API + GUI)

```bash
./run.sh
```

Starts Qdrant and vLLM (`docker compose up -d`), waits until both respond, then starts the API (http://127.0.0.1:8080, models load at startup) and the Streamlit GUI (http://localhost:8501). Ctrl+C stops the API and the GUI; the containers keep running (`docker compose down` stops them). The corpus must be indexed first (`uv run rag ingest --reset`, see Install).

In the GUI, log in by typing a user name (no password, see the users below). Admins (`ops`) also get a "Sync database" button in the sidebar.

Each part can also be started on its own:

```bash
uv run rag serve                           # API only
uv run streamlit run src/rag/ui/app.py     # GUI only; the API must be up. RAG_API_URL overrides the API address
```

### Example questions

| Log in as | Ask | Expect |
|-----------|-----|--------|
| `alice` | How many points did the Panthers defense surrender? | 308, with a citation |
| `alice` | What actor did sign language for the National Anthem at Superbowl 50? | Marlee Matlin |
| `alice` | Сколько очков уступила защита Пэнтерс? | 308, answered in Russian |
| `alice` | What year did Tesla die? then What was named after him in 1960? | 1943 (the current document, not the superseded one); the follow-up is rewritten using the first question; answer: the SI unit of magnetic flux density |
| `alice` | How many vacation days do new employees get per year? | `not_found`, no invented answer |
| `alice` | In what decade did ABC finish transitioning to color? | `not_found`: the document is restricted |
| `root` | In what decade did ABC finish transitioning to color? | 1970s |
| `ops` | (sidebar) Sync database | counts per category |

More questions with known answers are in [data/golden.jsonl](data/golden.jsonl), e.g. `grep '"kind": "acl"' data/golden.jsonl`.

### API (curl)

Ask a question (`-N` keeps the stream unbuffered):

```bash
curl -N -X POST localhost:8080/v1/ask \
  -H 'Content-Type: application/json' -H 'X-User: alice' \
  -d '{"question": "What year did Tesla die?"}'
```

The response is Server-Sent Events:

```
event: meta
data: {"conversation_id": "9f1c…", "query": "What year did Tesla die?"}

event: token
data: {"text": "Tesla died in 1943 [1]."}

event: result
data: {"status": "answered", "answer": "…", "citations": [{"n": 1, "chunk_id": "…", "doc_id": "…", "title": "…", "url": "…", "snippet": "…"}]}

event: done
data: {}
```

Follow-ups: pass the returned `conversation_id` back in the request body.

| Endpoint | Who | Purpose |
|----------|-----|---------|
| `POST /v1/ask` | any known user | Stream an answer |
| `POST /v1/admin/sync` | group `admin` (user `ops`) | Re-sync `data/corpus` into the index; returns counts per category |
| `GET /v1/admin/import/options` | admin | Groups, sources and authorities the import form offers |
| `POST /v1/admin/import` | admin | Download a web page into the corpus and sync (see below) |
| `GET /v1/me` | any known user | The caller's groups |
| `GET /health` | anyone | Liveness |

Users in [data/users.yaml](data/users.yaml): `alice` (all-employees), `bob` (+research), `hana` (+hr-team), `root` (research + hr-team, used by eval), `ops` (+admin). A document restricted to `hr-team` is invisible to `alice`: she gets `not_found`, not a leak.

**Importing a web page** (admin): in the GUI, open "Import a web page", enter a URL and the ACL groups, source and authority (title and language are optional overrides). [indexing/importer.py](src/rag/indexing/importer.py) downloads the text (Wikipedia via its API, other pages via trafilatura), keeps one paragraph per line, drops headings, fragments under 60 characters and repeated paragraphs, and skips paragraphs that are near-duplicates (cosine ≥ `postrank.dup_threshold`) of other indexed documents. It then writes `data/corpus/<doc_id>.json`, adds the entry to `meta.yaml` and syncs. The `doc_id` is derived from the URL, so importing the same URL again updates that document. The server downloads whatever URL the admin gives it, so keep this admin-only.

**Updating documents**: edit `data/corpus/*.json` or `meta.yaml`, then `curl -X POST localhost:8080/v1/admin/sync -H 'X-User: ops'` (or `uv run rag ingest`).

## Test

```bash
uv run pytest tests/unit                  # fast, no services, fakes only
uv run pytest -m integration              # needs Qdrant + real models; indexes the full corpus into a throwaway collection
uv run pytest -m llm                      # needs vLLM up; auto-skipped if unreachable
uv run pytest                             # everything
```

| Suite | Needs | Covers |
|-------|-------|--------|
| [tests/unit/](tests/unit/) | nothing | RRF, chunking, post-rank, prompt budget, citations, condensation, ACL/auth, API, pipeline fallbacks, InfoNCE, metrics, hashing |
| [tests/integration/](tests/integration/) | Qdrant + models | Real chunking, indexing/sync, retrieval, post-rank, Recall@5 floors per stage (regression guard) |
| [tests/e2e/](tests/e2e/) | + vLLM | Grounded cited answers, `NOT_FOUND`, follow-ups, no ACL leak through the API, LLM-down and dense-branch-down fallbacks |

## Evaluate

```bash
uv run rag eval                           # all golden questions
uv run rag eval --limit 200 --out run.json
uv run rag eval-generation                # writes data/eval/generation.jsonl for an LLM judge
```

`rag eval` reports Recall@5/10, MRR (Mean Reciprocal Rank) and NDCG@10 (Normalized Discounted Cumulative Gain) at every stage (dense, sparse, rrf, rerank, postrank) for six settings (EN→all, RU→all, EN→EN, RU→RU, EN→RU, RU→EN), then sweeps the gate threshold: share of answerable questions passed vs. unanswerable and ACL-protected questions blocked. Compare against [data/eval/baseline.json](data/eval/baseline.json) (base bge-m3).

`rag eval-generation` runs the real pipeline on the `generation`-tagged questions and stores each answer with its exact prompt and citations. Judge criteria: *faithful* (claims supported by cited passages), *correct* (agrees with reference answers), *citations* (each `[n]` supports its claim).

## Tune

Everything is in [config.yaml](config.yaml), with inline comments. Common knobs:

| Goal | Setting |
|------|---------|
| More/less "not found" | `rerank.gate_threshold` (re-run `rag eval` and read the gate table) |
| Trust dense vs. BM25 | `retrieval.rrf_weights` |
| Prefer fresh / official docs | `postrank.w_freshness`, `w_authority`, `half_life_days` |
| More context to the LLM | `postrank.final_k`, `max_per_doc` (mind `llm.context_window`) |
| Change chunk size | `chunking.*`, then `rag ingest` (changed hashes trigger re-embedding) |
| Change the LLM | `LLM_MODEL` env for compose, plus `llm.tokenizer` / `llm.context_window` |

### Fine-tuning the embedder

```bash
uv run rag train                          # symmetric InfoNCE + mined hard negatives → models/bge-m3-infonce
# set models.embedder to that directory in config.yaml, then:
uv run rag ingest --reset
uv run rag eval --out run.json            # compare with data/eval/baseline.json
```

Training data are SQuAD/SberQuAD train and MIRACL en/ru. SQuAD dev is deliberately excluded because XQuAD, the evaluation corpus, is drawn from it.

## Known limitations

- Conversation memory is in-process: lost on restart, not shared across workers, never expires.
- Auth is a toy `X-User` header; replace with real SSO (Single Sign-On) before any real use.
- Corpus and ACLs are synthetic, loaded from files rather than live connectors.
