from unittest.mock import patch

from app.models.retrieval import RetrievalResult
from app.retrieval.hybrid import HybridRetriever


@patch("app.retrieval.hybrid.DenseRetriever")
def test_hybrid_search_combines_results(mock_dense) -> None:
    mock_dense.return_value.search.return_value = [
        RetrievalResult(document_id="1",
            chunk_id="1",
            content="retrieval augmented generation",
            score=0.9,
            source="dense",
        ),
        RetrievalResult(
            document_id="2",
            chunk_id="2",
            content="python programming",
            score=0.8,
            source="dense",
        ),
    ]

    documents = [
    {
        "document_id": "1",
        "chunk_id": "1",
        "content": "python programming",
        "source": "test",
        "page": None,
    },
    {
        "document_id": "2",
        "chunk_id": "2",
        "content": "retrieval augmented generation",
        "source": "test",
        "page": None,
    },
]

    retriever = HybridRetriever(documents)
    results = retriever.search("retrieval generation", top_k=2)

    assert len(results) == 2
    assert all(result.source == "hybrid" for result in results)

    by_id = {result.document_id: result for result in results}

    assert by_id["1"].content == "python programming"
    assert by_id["2"].content == "retrieval augmented generation"


@patch("app.retrieval.hybrid.DenseRetriever")
def test_hybrid_can_select_bm25_without_calling_dense(mock_dense) -> None:
    documents = [
        {
            "document_id": "1",
            "chunk_id": "1",
            "content": "python programming",
            "source": "test",
        }
    ]

    retriever = HybridRetriever(documents)
    results = retriever.search("python", top_k=1, backend="bm25")

    assert results[0].source == "bm25"
    mock_dense.return_value.search.assert_not_called()