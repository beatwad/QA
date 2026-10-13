from typing import Protocol

from qdrant_client import models

from rag.lang import detect_lang

# detect_lang() code -> fastembed/Snowball stemmer language
STEMMER_LANGS = {"en": "english", "ru": "russian"}


class DenseEncoder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def encode_documents(self, texts: list[str], langs: list[str]) -> list[models.SparseVector]: ...
    def encode_query(self, text: str) -> models.SparseVector: ...


class BgeM3Encoder:
    """Dense bi-encoder: one shared vector space for EN and RU, L2-normalized."""

    def __init__(self, name: str = "BAAI/bge-m3", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = batch_size
        self.model = SentenceTransformer(name, model_kwargs={"torch_dtype": "float16"})
        self.dim = self.model.get_embedding_dimension()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]


class Bm25Encoder:
    """Sparse BM25 term weights with a per-language stemmer and stopwords.

    Only term frequencies are encoded here; Qdrant applies IDF at search time
    (sparse vector modifier), so IDF stays correct as the corpus changes.
    """

    def __init__(self):
        from fastembed.sparse.bm25 import Bm25

        self.models = {lang: Bm25("Qdrant/bm25", language=stem) for lang, stem in STEMMER_LANGS.items()}

    def encode_documents(self, texts: list[str], langs: list[str]) -> list[models.SparseVector]:
        return [self._convert(next(self.models[lang].embed([text]))) for text, lang in zip(texts, langs)]

    def encode_query(self, text: str) -> models.SparseVector:
        return self._convert(next(self.models[detect_lang(text)].query_embed(text)))

    @staticmethod
    def _convert(emb) -> models.SparseVector:
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
