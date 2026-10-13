from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class QdrantConfig(BaseModel):
    url: str
    collection: str


class LLMConfig(BaseModel):
    base_url: str
    model: str
    tokenizer: str
    context_window: int
    max_tokens: int
    temperature: float
    history_turns: int


class ModelsConfig(BaseModel):
    embedder: str
    reranker: str


class ChunkingConfig(BaseModel):
    max_tokens: int
    overlap: float


class RetrievalConfig(BaseModel):
    rrf_k: int
    rrf_weights: dict[str, float]
    candidates: int


class RerankConfig(BaseModel):
    top_k: int
    gate_threshold: float


class PostrankConfig(BaseModel):
    final_k: int
    max_per_doc: int
    w_freshness: float
    w_authority: float
    w_dup: float
    dup_threshold: float
    half_life_days: dict[str, float]
    authority_prior: dict[str, float]


class Config(BaseModel):
    qdrant: QdrantConfig
    llm: LLMConfig
    models: ModelsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    postrank: PostrankConfig


def load_config(path: Path = DEFAULT_PATH) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text()))
