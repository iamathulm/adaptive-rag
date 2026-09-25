from unittest.mock import MagicMock, patch

from app.retrieval import VectorStore


def test_vector_store_initializes() -> None:
    store = VectorStore()
    assert store.client is not None


@patch("app.retrieval.vector_store.QdrantClient")
def test_vector_store_search(mock_client) -> None:
    point = MagicMock()
    mock_client.return_value.query_points.return_value.points = [point]

    store = VectorStore()
    results = store.search([0.1] * 768, top_k=3)

    assert results == [point]
    mock_client.return_value.query_points.assert_called_once_with(
        collection_name="documents",
        query=[0.1] * 768,
        limit=3,
    )