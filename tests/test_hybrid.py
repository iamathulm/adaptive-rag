from unittest.mock import patch

from app.models.retrieval import RetrievalResult
from app.retrieval.hybrid import HybridRetriever


@patch("app.retrieval.hybrid.DenseRetriever")
def test_hybrid_search_combines_results(mock_dense) -> None:
    mock_dense.return_value.search.return_value = [
        RetrievalResult(
            document_id="2",
            content="retrieval augmented generation",
            score=0.9,
            source="dense",
        ),
        RetrievalResult(
            document_id="1",
            content="python programming",
            score=0.8,
            source="dense",
        ),
    ]

    documents = [
        "python programming",
        "retrieval augmented generation",
    ]

    retriever = HybridRetriever(documents)
    results = retriever.search("retrieval generation", top_k=2)

    assert len(results) == 2
    assert all(result.source == "hybrid" for result in results)

    by_content = {result.content: result for result in results}

    assert by_content["retrieval augmented generation"].document_id == "2"
    assert by_content["python programming"].document_id == "1"