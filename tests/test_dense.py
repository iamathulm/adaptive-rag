from unittest.mock import MagicMock, patch

from app.retrieval.dense import DenseRetriever


@patch("app.retrieval.dense.Embedder")
@patch("app.retrieval.dense.VectorStore")
def test_dense_search(mock_store, mock_embedder) -> None:
    mock_embedder.return_value.embed.return_value = [0.1] * 768

    point = MagicMock()
    point.id = 1
    point.payload = {"content": "test document"}
    point.score = 0.95

    mock_store.return_value.search.return_value = [point]

    retriever = DenseRetriever()
    results = retriever.search("test query", top_k=1)

    assert len(results) == 1
    assert results[0].content == "test document"
    assert results[0].score == 0.95
    assert results[0].source == "dense"