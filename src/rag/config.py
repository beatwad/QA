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


class PostrankConfig(BaseModel):
    final_k: int


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
