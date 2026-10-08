# Datasets

## Main corpus: XQuAD (English + Russian)

- Link: https://huggingface.co/datasets/google/xquad (configs `xquad.en`, `xquad.ru`)
- Paper: https://arxiv.org/abs/1910.11856
- License: CC BY-SA 4.0
- Size: 240 Wikipedia paragraphs, 1,190 questions per language, with answer spans.
- Fields: `id`, `context`, `question`, `answers {text, answer_start}`

Why it fits:
- **Parallel EN/RU:** the same paragraphs and questions exist in both languages, so we can test cross-lingual retrieval (a Russian question finding an English chunk, and the reverse) as well as same-language retrieval.
- **Labels come free:** each question's own `context` is its relevant passage, which gives Recall@k, MRR (Mean Reciprocal Rank) and NDCG (Normalized Discounted Cumulative Gain) without manual labeling.
- **Toy-sized:** 480 passages index in seconds, and each answer span serves as the reference answer.

Gaps and how we fill them:
- **No metadata:** the dataset has no ACL (Access Control List), `updated_at`, source, authority or status. We add synthetic metadata to the documents (some restricted, some superseded, some old) so the permission and post-ranking stages have something to act on.
- **No titles:** XQuAD's `id`s are the SQuAD v1.1 dev `id`s, so a join with SQuAD v1.1 dev (https://huggingface.co/datasets/rajpurkar/squad) recovers the article titles. The paragraphs can then be grouped into documents.
- **No unanswerable questions:** we add some by hand, and can also borrow questions about the same articles from SQuAD 2.0 (https://huggingface.co/datasets/rajpurkar/squad_v2).

## Training data for Phase 9 (bi-encoder fine-tuning with InfoNCE)

These sets don't overlap with the XQuAD eval questions, so there's no train/eval leakage:
- **SberQuAD**: Russian SQuAD-style QA, about 45k training pairs. https://huggingface.co/datasets/kuznetsoffandrey/sberquad. The license isn't stated on the card.
- **SQuAD v1.1 train**: English, about 87k pairs. https://huggingface.co/datasets/rajpurkar/squad. **Exclude the dev split** because XQuAD comes from it.
- **MIRACL `ru` / `en`**: queries with relevance judgments (qrels) that include judged non-relevant passages, which are useful as hard negatives. https://huggingface.co/datasets/miracl/miracl

## Alternatives considered

- **DRAGON:** a Russian-only RAG benchmark over news (https://arxiv.org/abs/2507.05713). No English, and part of the QA set is kept private.
- **RuBQ 2.0:** Russian knowledge-base questions with Wikipedia paragraphs (https://github.com/vladislavneon/RuBQ). The questions are mostly about Wikidata facts.
- **MIRACL as the main corpus:** the Wikipedia dumps are millions of passages, too large for a toy setup. It's kept for training only.
- **Synthetic "enterprise" corpora** (e.g. https://app.readytensor.ai/publications/aaidc-project-1-aether-dynamics-corp-internal-knowledge-corpus-oLpiX6rTUzwZ): closer to the business case, but English-only and without labeled relevance.
