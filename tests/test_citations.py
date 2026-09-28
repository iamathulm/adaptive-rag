from app.generation.citations import validate_citations
from app.models.retrieval import RetrievalResult


def test_validate_citations_deduplicates_known_chunks() -> None:
    result = RetrievalResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        content="context",
        score=0.9,
        source="sample.pdf",
    )

    citations, grounded = validate_citations("Answer [chunk-1] [chunk-1]", [result])

    assert [citation.model_dump() for citation in citations] == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source": "sample.pdf",
        }
    ]
    assert grounded is True


def test_validate_citations_rejects_unknown_chunks() -> None:
    result = RetrievalResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        content="context",
        score=0.9,
        source="sample.pdf",
    )

    citations, grounded = validate_citations("Answer [unknown]", [result])

    assert citations == []
    assert grounded is False