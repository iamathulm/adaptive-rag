from fastapi.testclient import TestClient

from app.main import app
from app.models.retrieval import RetrievalResult


def test_retrieval_endpoint() -> None:
    from app.api.dependencies import get_dense_retriever

    class FakeRetriever:
        def search(self, query: str, top_k: int | None = None):
            return [
                RetrievalResult(
                    document_id="1",
                    content=f"result for {query}",
                    score=0.95,
                    source="dense",
                )
            ]

    app.dependency_overrides[get_dense_retriever] = lambda: FakeRetriever()

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