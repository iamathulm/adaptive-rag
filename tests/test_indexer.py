from unittest.mock import patch

from app.retrieval.indexer import DocumentIndexer


@patch("app.retrieval.indexer.Embedder")
@patch("app.retrieval.indexer.VectorStore")
def test_index_documents(mock_store, mock_embedder) -> None:
    embedder = mock_embedder.return_value
    embedder.embed.return_value = [0.1] * 768

    store = mock_store.return_value

    indexer = DocumentIndexer()
    indexer.index([{"content": "test document", "source": "test"}])

    store.ensure_collection.assert_called_once_with("documents")
    store.client.upsert.assert_called_once()