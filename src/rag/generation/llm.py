from collections.abc import Iterator
from typing import Protocol


class LLM(Protocol):
    def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]: ...


def complete(llm: LLM, messages: list[dict], max_tokens: int, temperature: float = 0.0) -> str:
    """Non-streaming call: the whole answer as one string."""
    return "".join(llm.stream(messages, max_tokens, temperature))


class OpenAICompatLLM:
    """vLLM's OpenAI-compatible chat API, streamed token by token."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="unused", timeout=timeout, max_retries=0)
        self.model = model

    def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        """Read the streamed answer  piece by piece as vLLM generates it, 
        and passe on just the new text from each piece."""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
