import pytest

from app.evaluation import RetrievalEvaluator, load_evaluation_cases
from app.models.evaluation import EvaluationCase
from app.models.retrieval import RetrievalResult


class FakeRetriever:
    def search(self, query: str, top_k: int, backend: str):
        return [
            RetrievalResult(
                document_id="doc-1",
                chunk_id="chunk-2",
                content="second",
                score=0.9,
                source=backend,
            ),
            RetrievalResult(
                document_id="doc-1",
                chunk_id="chunk-3",
                content="third",
                score=0.8,
                source=backend,
            ),
        ]


def test_retrieval_evaluator_reports_metrics() -> None:
    evaluator = RetrievalEvaluator(FakeRetriever())
    results = evaluator.evaluate(
        [EvaluationCase(query="question", relevant_chunk_ids={"chunk-2", "chunk-4"})],
        top_k=2,
    )

    assert results[0].recall_at_k == 0.5
    assert results[0].precision_at_k == 0.5
    assert results[0].reciprocal_rank == 1.0
    assert results[0].ndcg_at_k == pytest.approx(0.613147)


def test_load_evaluation_cases(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"query": "question", "relevant_chunk_ids": ["chunk-1"], "backend": "bm25"}]',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert cases[0].backend == "bm25"
    assert cases[0].relevant_chunk_ids == {"chunk-1"}