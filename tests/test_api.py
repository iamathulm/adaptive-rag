from fastapi.testclient import TestClient

from app.main import app
from app.models.retrieval import RetrievalResult


def test_retrieval_endpoint() -> None:
    from app.api.dependencies import get_retriever

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return [
                RetrievalResult(
    document_id="1",
    chunk_id="1",
    content=f"result for {query}",
    score=0.95,
    source="dense",
)
            ]

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    try:
        client = TestClient(app)
        response = client.post(
            "/retrieval",
            json={"query": "test query", "top_k": 1},
        )

        assert response.status_code == 200
        assert response.json() == {
            "results": [
                {
                    "document_id": "1",
                    "content": "result for test query",
                    "score": 0.95,
                    "source": "dense",
                }
            ]
        }
    finally:
        app.dependency_overrides.clear()

def test_retrieval_endpoint_validates_request() -> None:
    client = TestClient(app)

    empty_query = client.post(
        "/retrieval",
        json={"query": ""},
    )
    assert empty_query.status_code == 422

    invalid_top_k = client.post(
        "/retrieval",
        json={"query": "test", "top_k": 0},
    )
    assert invalid_top_k.status_code == 422

def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_endpoint() -> None:
    from app.api.dependencies import get_generator, get_retriever

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return [
                RetrievalResult(
                    document_id="1",
                    chunk_id="chunk-1",
                    content=f"context for {query}",
                    score=0.95,
                    source="dense",
                )
            ]

    class FakeGenerator:
        def generate(self, query: str, results: list[RetrievalResult]) -> str:
            return f"answer for {query} [{results[0].chunk_id}]"

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()

    try:
        client = TestClient(app)
        response = client.post("/answer", json={"query": "test query", "top_k": 1})

        assert response.status_code == 200
        assert response.json() == {
            "answer": "answer for test query [chunk-1]",
            "sources": [
                {
                    "document_id": "1",
                    "chunk_id": "chunk-1",
                    "content": "context for test query",
                    "score": 0.95,
                    "source": "dense",
                }
            ],
        }
    finally:
        app.dependency_overrides.clear()


def test_answer_endpoint_validates_request() -> None:
    client = TestClient(app)

    empty_query = client.post("/answer", json={"query": ""})
    invalid_top_k = client.post("/answer", json={"query": "test", "top_k": 0})

    assert empty_query.status_code == 422
    assert invalid_top_k.status_code == 422


def test_answer_endpoint_hides_generation_failure() -> None:
    from app.api.dependencies import get_generator, get_retriever
    from app.generation.generator import GenerationError

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return []

    class FailingGenerator:
        def generate(self, query: str, results: list[RetrievalResult]) -> str:
            raise GenerationError("secret provider details")

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_generator] = lambda: FailingGenerator()

    try:
        client = TestClient(app)
        response = client.post("/answer", json={"query": "test"})

        assert response.status_code == 502
        assert response.json() == {
            "detail": {
                "code": "generation_failed",
                "message": "The answer provider could not generate a response",
            }
        }
        assert "secret provider details" not in response.text
    finally:
        app.dependency_overrides.clear()