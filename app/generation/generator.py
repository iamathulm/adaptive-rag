import json
import time
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import Settings, settings
from app.models.retrieval import RetrievalResult


class GenerationError(RuntimeError):
    """Raised when a generation provider cannot produce an answer."""


class AnswerGenerator(Protocol):
    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        ...


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    context = "\n\n".join(
        f"[{result.chunk_id}]\n{result.content}"
        for result in results
    )

    return f"""Answer the question using only the provided context.
If the context does not contain the answer, say that you do not have enough information.
Reference supporting chunks using their IDs in square brackets, for example [chunk-1].

Question:
{query}

Context:
{context}

Answer:"""


class MockAnswerGenerator:
    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        if not results:
            return "I do not have enough information to answer that."

        citations = " ".join(f"[{result.chunk_id}]" for result in results)
        return f"Mock answer for: {query} {citations}".strip()


class _HttpAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float,
        max_tokens: int,
        temperature: float,
        opener: Callable[..., object] = urlopen,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.opener = opener
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _request(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except HTTPError as exc:
                failure = exc
                retryable = exc.code == 429 or 500 <= exc.code <= 599
            except (URLError, TimeoutError) as exc:
                failure = exc
                retryable = True
            except json.JSONDecodeError as exc:
                failure = exc
                retryable = False

            if not retryable or attempt == self.max_retries:
                raise GenerationError("Generation provider request failed") from failure

            time.sleep(self.retry_backoff_seconds * (2**attempt))


class GeminiAnswerGenerator(_HttpAnswerGenerator):
    def generate(self, query: str, results: list[RetrievalResult]) -> str:
        payload = {
            "contents": [{"parts": [{"text": build_prompt(query, results)}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        response = self._request(url, payload, {"x-goog-api-key": self.api_key})

        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("Generation provider returned an invalid response") from exc


def create_generator(config: Settings = settings) -> AnswerGenerator:
    common = {
        "model": config.generation_model,
        "timeout": config.generation_timeout_seconds,
        "max_tokens": config.generation_max_tokens,
        "temperature": config.generation_temperature,
        "max_retries": config.generation_max_retries,
        "retry_backoff_seconds": config.generation_retry_backoff_seconds,
    }

    if config.generation_provider == "mock":
        return MockAnswerGenerator()
    if config.generation_provider == "gemini" and config.gemini_api_key:
        return GeminiAnswerGenerator(
            api_key=config.gemini_api_key.get_secret_value(),
            **common,
        )

    raise GenerationError(
        f"Provider '{config.generation_provider}' is unsupported or missing its API key"
    )