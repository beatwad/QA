# Internal Knowledge Q&A Bot (LLM + RAG)

## Business task

Employees waste time searching Confluence/Slack/Google Drive/wikis for answers, or interrupt SMEs (Subject Matter Experts) with questions that are already documented → reduce time-to-answer and reduce duplicate SME/support load.

## Functional requirements (FT)

- Given a natural language question, return a synthesized, grounded natural-language answer (not a list of links)
- Answer must be citable: every claim is backed by a link + snippet to the source document
- If the answer is not in the indexed corpus, say so explicitly — no hallucinated answers
- Respect document permissions: a user must never see an answer that leaks content from a document they don't have access to
- Multi-turn conversation: follow-up questions can refer to previous turns
- Multiple sources: Confluence, Slack, Google Drive, internal wiki, ticket system

## Non-functional requirements (NFT)

- 5 million source documents → after chunking ~20 million chunks
- 10,000 employees, 3,000 DAU (Daily Active Users), ~5 questions/user/day → ~15,000 queries/day → ~0.5 QPS (Queries Per Second) average, ~5 QPS peak
- Freshness: edited/deleted doc or permission change must be reflected in answers within ~15 minutes
- Latency target is split into two numbers, because generation is streamed: p99 time-to-first-token < 1.5 s, p99 time-to-full-answer < 10 s

## ML task

Two chained ML tasks over a private, permission-gated, constantly-changing corpus:

1. **Retrieval** (ranking task) — given a query, return the top-k most relevant chunks the user is allowed to see
2. **Grounded generation** (conditional language modeling) — given the query + retrieved chunks, generate an answer that is entailed by those chunks, with citations

## Data

- Source document: raw text/HTML/markdown/PDF content, source system, url, title, author, timestamp created/updated, space/workspace, **ACL** (Access Control List - list of groups/users allowed to view it)
- Chunk (the actual retrieval unit): chunk_id, parent doc_id, chunk text, position in doc, embedding, ACL inherited from parent doc
- Query logs: user_id, query text, conversation_id, timestamp, retrieved chunk_ids + rerank scores, generated answer, shown citations, thumbs up/down, follow-up query in same session
- Training pairs for the retriever: (query, relevant_chunk) — bootstrapped two ways:
    - synthetic: prompt an LLM (Large Language Model) to generate plausible questions for a chunk (doc2query-style), and add them to the chunk's indexed text — this closes the vocabulary mismatch between document wording and real user queries
    - real: clicked citation in a thumbs-up answer = positive; chunk retrieved but not clicked/not cited = candidate negative
- Golden eval set: SME-curated (question, correct chunk_ids, reference answer), stratified by department/domain, refreshed regularly since correct answers can go stale as docs change

## Model

### Why RAG (Retrieval-Augmented Generation) instead of just an LLM, or a fine-tuned LLM

- A pretrained LLM knows nothing about this company's internal docs (never in training data) and its knowledge is frozen at a training cutoff — internal docs change hourly
- Fine-tuning the LLM's weights on internal docs to "bake in" this knowledge doesn't work well here: docs and permissions change constantly (retraining can't keep up), the model can't say *which* document a fact came from (no citations), and it can't restrict what a specific user is allowed to know (permissions live outside the weights)
- RAG instead keeps knowledge in an external, cheaply-updatable index. At query time we retrieve the relevant text and put it directly into the LLM's context window — the model only has to *read and synthesize what's put in front of it*, not recall it from memory. This is far more accurate, immediately reflects edits/deletes/permission changes, and gives citations for free (we know exactly which chunk supported which sentence)
- Trade-off: answer quality is bounded by retrieval quality — if the right chunk is never retrieved, no LLM can answer correctly, it will either say "I don't know" (fine) or hallucinate (bad). This is why most of the design effort below goes into the retrieval/ranking pipeline; the LLM itself is treated as a commodity component

### Chunking

Documents are split into chunks before indexing, because: the LLM context window is finite (can't stuff a whole wiki space into one prompt), and embedding models produce a much more useful vector for a short, focused passage than for a long document (embedding a 5,000-word page collapses many different facts into one blurry vector — the query "what's our vacation policy" gets diluted by every other topic on the page).

- Target ~300 tokens/chunk, ~15% overlap between adjacent chunks — overlap exists so a fact that straddles a chunk boundary isn't lost entirely to either chunk
- Chunk boundaries respect document structure (headings/sections), not arbitrary token cuts — cutting mid-sentence or mid-table destroys retrievability
- Chunk size is a trade-off: too small → embedding has too little context to be distinctive and answers need many chunks stitched together; too big → embedding gets diluted (same problem as not chunking at all) and wastes context-window budget on irrelevant text sent to the LLM

### Retrieval — hybrid dense + lexical

- Dense branch: bi-encoder embeds query and chunk into the same vector space, ANN (Approximate Nearest Neighbor)/HNSW (Hierarchical Navigable Small World, a graph-based ANN algorithm) search over the chunk index. Good for paraphrased/semantic questions ("how do I get reimbursed" ↔ doc titled "Expense policy")
- Lexical branch (BM25/Elasticsearch — BM25 is a classic keyword-matching ranking algorithm): good for exact terms embeddings blur away — error codes, ticket numbers, internal acronyms, people's names
- Both branches are fused with **Reciprocal Rank Fusion (RRF)**:

    ```
    score = Σ_lists w_l / (k + rank_i), k ≈ 60
    ```

    Ranks are used instead of raw scores because cosine similarity (0..1) and BM25 (unbounded, corpus-dependent) aren't on comparable scales and both drift as the index changes. `k` flattens the top, so appearing in **both** lists is worth more than being rank 1 in only one
- **ACL filter is applied inside both searches, not after** — a user with narrow permissions could otherwise get an empty top-k after post-filtering. Missing/unresolved ACL metadata **fails closed** (chunk excluded, never shown) — this is a hard security requirement, not a UX tradeoff

### Reranker

Cross-encoder (query + chunk in one sequence, a MiniLM-class model) reranks the fused top ~50 candidates down to a top ~20 shortlist (post-ranking below cuts that to the 5–10 chunks that actually get sent to the LLM). This step matters *more* here than in plain search: irrelevant chunks in the LLM's context don't just lower a ranking metric, they directly cause hallucination or a distracted/wrong answer ("lost in the middle" effect), so precision at very small k is the priority.

### Post-ranking (business-signal re-scoring after the cross-encoder)

The cross-encoder scores *topical relevance only*. It cannot tell a current policy page from its 2019 archived copy, or an official HR document from a Slack guess by an intern that happens to use the same words — both look equally relevant to the query. But whatever chunk wins here is what the LLM will assert, with a citation, as company truth. So the reranked top ~20 goes through one more cheap re-scoring pass before the final 5-10 chunks are cut.

Signals:

- **Freshness** — exponential decay on the doc's last-substantive-update timestamp, with a **per-source-type half-life** rather than one global constant: a 3-year-old vacation policy is fine, a 3-year-old deploy runbook is dangerous
- **Authority / source prior** — official Confluence space or approved policy doc > ticket comment > ad-hoc Slack message. Plus per-doc signals: SME-verified/endorsed flag, view count, number of internal inbound links (PageRank-ish "everyone links to this page")
- **Lifecycle flags** — `archived` / `deprecated` / `superseded_by` metadata → hard demotion or outright exclusion, not a soft penalty. Cheapest signal to maintain and the highest-value one, because a superseded doc is not "slightly less relevant", it's actively wrong
- **Diversity / dedup** — near-duplicates (the same policy copy-pasted into three spaces, or two overlapping adjacent chunks) burn the tiny context budget and give the LLM false corroboration of a single source. MMR-style (Maximal Marginal Relevance) selection plus a cap of ~2-3 chunks per parent doc, so one long document can't occupy every slot

Combination — start with a hand-tuned, relevance-dominant blend over normalized signals:

```
final = s_ce_norm + w_f * freshness + w_a * authority - w_d * dup_penalty
```

`s_ce_norm` is the sigmoid/min-max-normalized cross-encoder score, and the business weights are deliberately kept small: they should reorder near-ties (two chunks that both answer the question — prefer the fresh, authoritative one) but must never promote an off-topic chunk above a clearly relevant one. Once enough citation-click/thumbs data exists, replace the hand-tuned weights with a small **LTR (Learning to Rank)** model — GBDT (Gradient Boosted Decision Trees)/LambdaMART over `[cross-encoder score, RRF score, BM25 score, freshness, authority, historical citation-CTR of the chunk]`. The feature set is small and interpretable, so this stays ~1 ms for 20 candidates and stays debuggable.

The final ordering also decides prompt slot order, not just membership: the strongest chunk goes at the start or the end of the context block, never buried in the middle ("lost in the middle").

Caveats:

- **Freshness must be query-dependent** — "what *was* our 2023 travel policy" is a legitimate historical question; the recency boost is suppressed when query understanding detects an explicit time reference
- **Timestamp hygiene** — bulk migrations or format cleanups touch `updated_at` on every doc at once and make the whole corpus look fresh. Track a content-hash-based last-substantive-change instead of the connector's raw mtime
- **The confidence gate stays on the raw cross-encoder score**, not on the post-ranked score — a chunk boosted for being fresh and authoritative is not evidence that the question is answerable at all
- **Popularity/CTR features are a feedback loop** — a chunk that gets cited gets boosted, gets cited more. Cap their weight and monitor the concentration of citations over a small set of docs

### Query understanding

- Multi-turn condensation: a raw follow-up like "what about the timeout setting?" is meaningless to an embedding model on its own. Before retrieval, a small/fast LLM call rewrites (query + last few turns) into one standalone query
- Confidence gate: if the top reranked score is below a threshold, skip generation entirely and return "I couldn't find this in our docs, try asking #<team-channel>" — cheaper and safer than letting the LLM try to answer from thin context

### Generation (LLM)

- Prompt = system instructions ("answer only using the provided context; if the answer isn't there, say so; cite sources as [1], [2]…") + the 5–10 retrieved chunks (each tagged with its source title/url) + condensed question + recent conversation turns
- Token budget is actively managed: chunks + history + question must fit under the context window with room left for the answer; older turns get summarized rather than dropped silently
- Streamed output, so the user sees the first tokens quickly even though the full answer takes longer
- Model choice is a build-vs-buy decision: self-hosted open-weight model (full control, data never leaves the network, fixed infra cost) vs. hosted enterprise API with a zero-data-retention agreement (no GPU ops, pay per token). Given the low QPS here (~15k queries/day) API cost is often cheaper than running a dedicated GPU cluster — at much higher query volumes the calculus flips, since per-token API cost scales linearly with traffic while a self-hosted fleet's fixed cost gets amortized

### Grounding / hallucination mitigation

- Prompt-level constraint ("only use provided context") reduces but does not eliminate hallucination
- Post-hoc groundedness check: an NLI (Natural Language Inference) model or a cheap LLM-judge verifies each generated sentence is entailed by the cited chunk. Too slow/expensive to block the user's response, so it runs **async** on a sample and feeds the monitoring/eval pipeline, not the live request

### RAG failure taxonomy (useful for debugging any bad answer)

A wrong or unhelpful answer can come from four different places, and they need different fixes:

1. **Retrieval failure** — the relevant chunk was never retrieved at all (bad recall)
2. **Ranking failure** — it was retrieved but reranked out of the top-k sent to the LLM
3. **Generation/grounding failure** — it *was* in the context, but the LLM ignored it or misread it and hallucinated anyway
4. **Citation failure** — the answer is correct and grounded, but points to the wrong (or no) source

This is why RAG evaluation needs several separate metrics instead of one end-to-end score — Recall@k catches (1)/(2), faithfulness/groundedness catches (3), citation accuracy catches (4).

## Loss function

Retriever is a bi-encoder trained with a symmetric InfoNCE contrastive loss:

- `q_i`, `c_i` — L2-normalized embeddings of query i and chunk i in a batch of size N
- `sim(q, c)` = cosine similarity = dot product of normalized embeddings
- `temp` — learnable temperature (init ~0.07), kept in log-scale and clamped

```
query -> chunk: L_q2c(i) = -log( exp(sim(q_i, c_i)/temp) / sum_k exp(sim(q_i, c_k)/temp) )
chunk -> query: L_c2q(i) = -log( exp(sim(c_i, q_i)/temp) / sum_k exp(sim(c_i, q_k)/temp) )

L = 1/(2N) * sum_i (L_q2c(i) + L_c2q(i))
```

`i, i` is the positive pair, all other pairs in the batch are negatives. The sum in the denominator runs over all `k = 1..N` (positive included), otherwise it isn't a valid softmax over the batch.

Reranker (cross-encoder) is trained pointwise with binary cross-entropy on labeled (query, chunk, relevant?) pairs, or listwise if graded relevance labels are available.

The generator LLM itself is **not trained from scratch** and usually isn't trained at all — it's a pretrained instruction-tuned model used purely via prompting (in-context learning). Optional light SFT (Supervised Fine-Tuning; cross-entropy next-token loss on curated (context, question, cited answer) triples), or LoRA fine-tuning, only to fix citation formatting/tone — never to inject factual knowledge, since that's what retrieval is for.

## Offline metrics

- Retrieval: Recall@k, NDCG@k (Normalized Discounted Cumulative Gain), MRR (Mean Reciprocal Rank) on the golden (query → relevant chunk_ids) set
- Faithfulness/groundedness: fraction of generated claims entailed by the cited chunks (NLI model or LLM-as-judge)
- Answer relevancy: does the answer actually address the question (LLM-judge, or embedding similarity to a reference answer)
- Context precision/recall (RAGAS-style — RAGAS is a common RAG evaluation framework): of the chunks sent to the LLM, how many were actually used vs. noise (precision); of the chunks needed to fully answer, how many were retrieved (recall)
- Citation accuracy: does the cited source actually contain the claim it's attached to
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)/BLEU (Bilingual Evaluation Understudy) — both are text-overlap metrics — against a reference answer, only as a coarse regression guard — weak signal for open-ended answers that admit many valid phrasings, never the primary metric

## Online metrics

- Thumbs up/down rate on the answer
- Reformulation rate — user re-asks a rephrased version of the same question in the same session, a stronger dissatisfaction signal than a thumbs-down alone, since it's the user explicitly acting on an unresolved need rather than passively rating
- Escalation rate — user opens a support ticket / pings a human after the bot answered → strongest negative signal, means the bot actively failed to resolve
- Citation click-through rate (CTR) — needs care to interpret: low CTR + high thumbs-up is good (answer was self-sufficient); high CTR + thumbs-down usually means wrong sources were surfaced
- Deflection rate (the actual business metric) — measured reduction in support tickets/SME interruptions attributable to the bot being available

## Train

- Build the SME-curated golden eval set **first** — nothing else can be judged without it, and it needs periodic refresh as source docs change underneath it
- Train the retriever bi-encoder with InfoNCE + hard-negative mining from ANN top-k (chunks the current retriever scores highly but that aren't actually relevant — "hard" because the model is confidently wrong on them)
- Train the cross-encoder reranker on labeled pairs, optionally distill back into the bi-encoder, repeat for 2-3 rounds until it stops improving
- Feedback flywheel: thumbs-down and escalations go to human review (raw thumbs-down alone is **not** a safe hard-negative label — it could mean wrong chunk retrieved, or right chunk with a badly phrased answer, and only SME review can tell which), reviewed cases feed back into the golden set and hard-negative pool
- Optional LoRA fine-tune of the generator on curated (context, question, cited answer) examples, purely for citation formatting/tone consistency, never for facts

## Inference

### Indexing pipeline (offline/continuous, triggered per doc change — not per query)

- Connectors pull from Confluence/Slack/Google Drive (webhook where available, periodic sync otherwise), detect create/update/delete
- Extract text (strip HTML/markdown, OCR (Optical Character Recognition) for scanned PDFs/images)
- Chunk (semantic, ~300 tokens, 15% overlap, heading-aware)
- Embed each chunk with the bi-encoder, store in Vector Index + BM index, with ACL resolved from the parent doc
- Permission-only change → cheap metadata update, no re-embedding
- Doc deleted → tombstone/remove its chunks
- Index freshness lag (source edit timestamp → searchable timestamp) is tracked as an SLA (Service-Level Agreement), target ~15 min

### Serving pipeline (online, per query)

1. Authenticate user, resolve their permission groups
2. If multi-turn: condense (query + recent history) into a standalone query
3. Embed query + extract keywords
4. Hybrid retrieval: ANN dense + BM25 lexical, both ACL-filtered inside the search
5. RRF-fuse → top ~50 candidates
6. Cross-encoder rerank → top ~20 chunks
7. Post-rank: blend in freshness/authority/lifecycle signals, drop near-duplicates and cap chunks per parent doc → final top 5–10
8. Confidence gate (on the raw cross-encoder score): below threshold → return "not found" fallback, skip the LLM call entirely
9. Assemble prompt (system instructions + chunks with source tags, strongest chunk first/last + condensed question + recent turns)
10. Call LLM, stream tokens back to the user
11. Map citation markers in the output back to chunk source metadata (title, url, snippet)
12. Async, non-blocking: sample the exchange into the groundedness/eval pipeline
13. Log feedback affordances (thumbs, citation clicks) for the training flywheel

Use Document Update Service and Chunk Encoder to load new document from Outer Sources, split them by chunks, encode them and add to Vector Index

### Notes

Text preprocessing (chunking, tokenization) must be identical between the indexing pipeline and query-time preprocessing to avoid train/serve skew.
Embedding model version and vector index version must be tracked together — a mismatch makes cosine similarity meaningless, so this needs a hard alert and a deploy block, not just a dashboard.

## A/B tests

- Randomization unit: user
- Ladder, cheap → expensive:
    1. Offline: Recall@k/NDCG on golden set + faithfulness score
    2. Online A/B: 1-5% of traffic, primary metric = thumbs-up rate / deflection rate, guardrails = escalation rate and a sampled human-reviewed hallucination rate (automatic groundedness score isn't trusted alone for the ship/no-ship call)
    3. Ramp-up with guardrails checked at each step
- Off-policy evaluation and interleaving (useful for systems that show a ranked list of results) don't transfer well here — there's one synthesized answer per query, not a list the user scans, so there's nothing to interleave and no logged ranking policy to reweight
- Report metrics per department/team, not only aggregated — a regression for one team (e.g. legal, whose docs are sparse) easily hides inside a good aggregate number

## Monitoring

- Retrieval quality proxies: ANN recall vs. brute force, share of results contributed by each branch (dense vs. lexical) — RRF can mask a dead branch, since the fused list still looks full while half the recall is silently gone
- Index freshness lag (doc edit → searchable)
- Age and authority distribution of *cited* docs — share of answers citing a doc older than N months or flagged archived/deprecated; drift here means the post-ranking weights (or the lifecycle metadata feeding them) have gone stale
- Groundedness/faithfulness score sampled over time, alert on spike (generation-side regression, e.g. after an LLM/prompt version change)
- "I don't know" rate — too high signals index/coverage gaps, too low may mean the confidence gate is too permissive and the model is guessing instead of declining
- Citation click-through and thumbs-down rate, broken out per team/department — surfaces systematic content gaps for specific domains
- Token usage / cost per query drift — generation cost is the dominant infra cost here, unlike a pure search system
- **Permission-leak canary**: scheduled automated queries from synthetic low-permission test accounts, asserting that restricted documents never appear in retrieved chunks or citations — this is a silent-failure class unique to this system (a leak produces no error, just a wrong/dangerous answer) and needs a hard alert, not a dashboard

## Fallback (graceful degradation)

- LLM generation unavailable → return the raw top retrieved chunks as ranked links ("couldn't summarize, but here are the most relevant docs")
- Retrieval branch down (dense or lexical) → serve from the remaining branch (dense-only hurts tail/exact-term queries most, lexical-only hurts paraphrased/semantic queries most)
- Low confidence retrieval → explicit "not found, try #channel" instead of letting the LLM attempt an answer from weak context
- Permission service unavailable → **fail closed**: refuse to answer rather than risk exposing restricted content — this is the one place where "serve something worse" is not acceptable, unlike every other fallback above
- Query condensation step fails → fall back to using the raw follow-up query for retrieval (degraded recall on ambiguous follow-ups, but still functional)
- Groundedness checker unavailable → answer still ships to the user (it's async/non-blocking by design), but the exchange is flagged as unverified in the monitoring pipeline

## Latency estimation

Unlike a search system, the meaningful p99 isn't one number — it's split into time-to-first-token (perceived latency) and time-to-full-answer (streamed, so users start reading before it's done).

- Auth/permission lookup — 5 ms
- Query condensation (only when multi-turn): small/fast LLM call — +300-400 ms, conditional
- Query embedding (batch=1, memory-bandwidth bound rather than compute bound at this batch size) — ~5 ms
- Hybrid retrieval over ~20M chunks (dense ANN + BM25 in parallel) — ~20 ms
- Cross-encoder rerank of top 50 (MiniLM-L6: 6 layers * 384 hidden → ~10.6M non-embedding params). Compute time is estimated from FLOP (floating-point operations, the amount of math the model needs to do) divided by achievable throughput, where MFU (Model FLOP Utilization — the fraction of the hardware's theoretical peak compute actually achieved in practice) accounts for the gap between peak and real-world speed:

    ```
    FLOP = 2 * 10.6*10^6 * (50 candidates * 256 tokens) = 2.7*10^11
    t = 2.7*10^11 / (3.12*10^14 * 0.1 MFU) ~ 9 ms
    ```

- Post-ranking (metadata feature lookup + blend/dedup over 20 candidates, no neural net) — ~1 ms
- Prompt assembly — 5 ms
- LLM prefill (time-to-first-token contribution): self-hosted 70B model, ~3,250 prompt tokens (system + 8 chunks*300 tok + history), tensor-parallel across 8 A100s:

    ```
    FLOP = 2 * 70*10^9 * 3250 = 4.55*10^14
    t = 4.55*10^14 / (8 * 3.12*10^14 * 0.4 MFU) ~ 450 ms
    ```

- **Time-to-first-token ≈ 5+5+20+9+1+5+450 ≈ 495 ms** (budget is 1.5 s p99 → comfortable headroom)
- LLM decode (streamed, doesn't block perceived latency but bounds full-answer time): memory-bandwidth bound — reading 70B params (70 GB fp8) across 8 GPUs' aggregate ~16 TB/s HBM (High Bandwidth Memory, the GPU's on-chip memory) bandwidth per token:

    ```
    t/token = 140 GB / 16 TB/s ~ 8.75 ms/token
    300-token answer: 300 * 8.75 ms ~ 2.6 s
    ```

- **Time-to-full-answer ≈ 495 ms + 2.6 s ≈ 3.1 s** (budget is 8 s p99 → headroom for queueing, cold cache, multi-turn condensation add-on)

## Memory estimation

- Vector index: 20M chunks * 768-dim * 2 bytes (fp16) = ~31 GB raw. PQ-compressed (PQ = Product Quantization; 96 sub-vectors, 1 byte each) → 20M * 96 bytes ≈ 1.9 GB — fits comfortably on a single machine
- Lexical index: 20M chunks * ~1.2 KB text ≈ 24 GB raw text, inverted index of similar order — a handful of nodes is enough at this corpus size
- LLM weights: 70 B params, 1 byte/param -> 70 GB + KV-cache (40 layers * 5000 tokens * 8 KV heads * 128 head dim = 2*10^5 * 10^3 = 2\*10^8) ~ 71 GB, 8 A100s have 640 GB RAM - acceptable

## Compute estimation (GPU fleet)

- At ~5 QPS peak, the retriever/reranker fleet is tiny — retrieval is not the compute bottleneck here
- The LLM decode fleet is: each request occupies a decode "slot" for ~2.6 s of streaming. At 5 QPS with ~3 s average request lifetime, peak concurrency ≈ 15 simultaneous in-flight requests. With continuous batching, a single 8-GPU node serving a 70B model typically handles tens of concurrent decode streams, so **one node (plus a second for HA (High Availability)/failover) covers peak load** — decode throughput is shared across concurrent requests via batching, rather than each request needing its own dedicated GPU-bound call, which is what would force the fleet to scale linearly with QPS
- Given the low absolute query volume (~15k/day), a hosted LLM API (with a zero-data-retention/enterprise agreement) is frequently cheaper in total cost of ownership than operating this GPU fleet — worth evaluating explicitly as build-vs-buy before committing to self-hosting
