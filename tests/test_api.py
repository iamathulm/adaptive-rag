from pathlib import Path

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
    assert response.headers["x-request-id"]


def test_request_id_is_preserved() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"x-request-id": "request-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"


def test_liveness_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint_reports_dependencies(monkeypatch) -> None:
    import app.main as main_module

    class FakeClient:
        def get_collections(self):
            return object()

    class FakeVectorStore:
        client = FakeClient()

    monkeypatch.setattr(main_module, "VectorStore", FakeVectorStore)
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"qdrant": "ok", "generation": "ok"},
    }


def test_readiness_endpoint_hides_qdrant_failure(monkeypatch) -> None:
    import app.main as main_module

    class FailingClient:
        def get_collections(self):
            raise RuntimeError("private connection details")

    class FailingVectorStore:
        client = FailingClient()

    monkeypatch.setattr(main_module, "VectorStore", FailingVectorStore)
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "checks": {"qdrant": "unavailable", "generation": "ok"},
    }
    assert "private connection details" not in response.text


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
            "citations": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "1",
                    "source": "dense",
                }
            ],
            "grounded": True,
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


def test_answer_endpoint_marks_fabricated_citations_as_ungrounded() -> None:
    from app.api.dependencies import get_generator, get_retriever

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return [
                RetrievalResult(
                    document_id="1",
                    chunk_id="chunk-1",
                    content="context",
                    score=0.95,
                    source="dense",
                )
            ]

    class FakeGenerator:
        def generate(self, query: str, results: list[RetrievalResult]) -> str:
            return "Unsupported claim [not-a-result]"

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()

    try:
        client = TestClient(app)
        response = client.post("/answer", json={"query": "test"})

        assert response.status_code == 200
        assert response.json()["citations"] == []
        assert response.json()["grounded"] is False
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


def test_answer_endpoint_returns_rate_limit_response() -> None:
    from app.api.dependencies import get_generator, get_retriever
    from app.generation.limiter import GenerationRateLimitError

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return []

    class RateLimitedGenerator:
        def generate(self, query: str, results: list[RetrievalResult]) -> str:
            raise GenerationRateLimitError(7)

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_generator] = lambda: RateLimitedGenerator()

    try:
        client = TestClient(app)
        response = client.post("/answer", json={"query": "test"})

        assert response.status_code == 429
        assert response.headers["retry-after"] == "7"
        assert response.json() == {
            "detail": {
                "code": "generation_rate_limited",
                "message": "The generation request limit has been reached",
            }
        }
    finally:
        app.dependency_overrides.clear()


def test_ingest_endpoint_indexes_uploaded_pdf() -> None:
    from app.api.dependencies import get_ingestion_service

    captured = {}

    class FakeIngestionService:
        def ingest(self, file_path: str, document_id: str) -> int:
            captured["path"] = file_path
            captured["document_id"] = document_id
            with open(file_path, "rb") as uploaded_file:
                captured["content"] = uploaded_file.read()
            return 3

    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()

    try:
        client = TestClient(app)
        response = client.post(
            "/ingest",
            files={"file": ("sample.pdf", b"pdf bytes", "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json() == {"document_id": "sample", "chunks_indexed": 3}
        assert captured["document_id"] == "sample"
        assert captured["content"] == b"pdf bytes"
        assert not Path(captured["path"]).exists()
    finally:
        app.dependency_overrides.clear()


def test_ingest_endpoint_rejects_unsupported_files() -> None:
    from app.api.dependencies import get_ingestion_service

    class FailingIngestionService:
        def ingest(self, file_path: str, document_id: str) -> int:
            raise AssertionError("unsupported files must be rejected before ingestion")

    app.dependency_overrides[get_ingestion_service] = lambda: FailingIngestionService()

    try:
        client = TestClient(app)
        response = client.post(
            "/ingest",
            files={"file": ("sample.txt", b"text", "text/plain")},
        )

        assert response.status_code == 415
        assert response.json() == {"detail": "Only PDF and DOCX files are supported"}
    finally:
        app.dependency_overrides.clear()