# app/evaluation.py

from app.models.evaluation import EvaluationCase, EvaluationResult
from app.retrieval.hybrid import HybridRetriever


class RetrievalEvaluator:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
    ) -> list[EvaluationResult]:
        results = []

        for case in cases:
            retrieved = self.retriever.search(
                case.query,
                top_k=top_k,
            )

            retrieved_ids = [
                result.chunk_id
                for result in retrieved
            ]

            relevant = case.relevant_chunk_ids
            hits = set(retrieved_ids) & relevant

            recall = 1.0 if hits else 0.0

            reciprocal_rank = 0.0
            for rank, document_id in enumerate(retrieved_ids, start=1):
                if document_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            results.append(
                EvaluationResult(
                    query=case.query,
                    retrieved_chunk_ids=retrieved_ids,
                    recall_at_k=recall,
                    reciprocal_rank=reciprocal_rank,
                )
            )

        return results