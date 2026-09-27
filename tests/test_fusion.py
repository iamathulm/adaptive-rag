from app.models.retrieval import RetrievalResult
from app.retrieval.fusion import RRFFusion


def test_rrf_fuses_results_by_document_id() -> None:
    first = RetrievalResult(
    document_id="1",
    chunk_id="1",
    content="first document",
    score=0.9,
    source="dense",
)
    second = RetrievalResult(
    document_id="2",
    chunk_id="1",
    content="second document",
    score=0.8,
    source="dense",
)

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            [first, second],
            [second, first],
        ],
        top_k=2,
    )

    assert [result.document_id for result, _ in results] == ["1", "2"]