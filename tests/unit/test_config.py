import pytest
from pydantic import ValidationError

from rag.config import load_config


def test_loads_repo_config(config):
    assert config.retrieval.rrf_k == 60
    assert config.rerank.top_k <= config.retrieval.candidates
    assert config.postrank.final_k <= config.rerank.top_k


def test_missing_section_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("qdrant: {url: x, collection: y}\n")
    with pytest.raises(ValidationError):
        load_config(path)
