from unittest.mock import patch

from app.models.chunk import DocumentChunk
from app.retrieval.indexer import DocumentIndexer


@patch("app.retrieval.indexer.Embedder")
@patch("app.retrieval.indexer.VectorStore")
def test_index_documents(mock_store, mock_embedder) -> None:
    embedder = mock_embedder.return_value
    embedder.embed.return_value = [0.1] * 768

    store = mock_store.return_value

    indexer = DocumentIndexer()

    chunk = DocumentChunk(
        document_id="doc-1",
        chunk_id="doc-1:0",
        content="test document",
        source="test",
        page=1,
    )

    indexer.index([chunk])

    store.ensure_collection.assert_called_once_with("documents")
    store.client.upsert.assert_called_once()

    call = store.client.upsert.call_args
    points = call.kwargs["points"]

    assert len(points) == 1
    assert points[0]["vector"] == [0.1] * 768
    assert points[0]["payload"] == {
        "document_id": "doc-1",
        "chunk_id": "doc-1:0",
        "content": "test document",
        "source": "test",
        "page": 1,
    }