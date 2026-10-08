import re

import httpx
import pytest

pytestmark = pytest.mark.llm

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def _complete(config, prompt: str) -> str:
    resp = httpx.post(
        config.llm.base_url + "/chat/completions",
        json={
            "model": config.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def test_english_completion(config):
    answer = _complete(config, "What is the capital of France? Answer in one word.")
    assert "paris" in answer.lower()


def test_russian_completion(config):
    answer = _complete(config, "Какая столица Франции? Ответь одним словом.")
    assert CYRILLIC.search(answer)
    assert "париж" in answer.lower()
