import json

import pytest

from app.core.config import Settings
from app.generation.generator import (
    GeminiAnswerGenerator,
    GenerationError,
    MockAnswerGenerator,
    create_generator,
)
from app.models.retrieval import RetrievalResult

RESULTS = [
    RetrievalResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        content="The answer is 42.",
        score=0.9,
        source="sample.txt",
    )
]


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_mock_generator_is_deterministic() -> None:
    generator = MockAnswerGenerator()

    assert "chunk-1" in generator.generate("What is the answer?", RESULTS)
    assert generator.generate("Unknown", []) == "I do not have enough information to answer that."


def test_gemini_generator_parses_response() -> None:
    requests = []

    def open_request(request, **_kwargs):
        requests.append(request)
        return FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "Grounded answer [chunk-1]"}]}}]}
        )

    generator = GeminiAnswerGenerator(
        api_key="test-key",
        model="test-model",
        timeout=1,
        max_tokens=10,
        temperature=0,
        opener=open_request,
    )

    assert generator.generate("Question", RESULTS) == "Grounded answer [chunk-1]"
    assert len(requests) == 1
    assert "test-key" not in requests[0].full_url
    assert requests[0].get_header("X-goog-api-key") == "test-key"


def test_create_generator_rejects_missing_provider_key() -> None:
    with pytest.raises(GenerationError, match="missing its API key"):
        create_generator(Settings(generation_provider="gemini"))


def test_create_generator_uses_mock_by_default() -> None:
    assert isinstance(create_generator(Settings()), MockAnswerGenerator)