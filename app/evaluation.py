import json
import math
from pathlib import Path

from app.models.evaluation import EvaluationCase, EvaluationResult
from app.retrieval.hybrid import HybridRetriever


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(case) for case in data]


def _ndcg(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), len(retrieved_ids))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


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
                backend=case.backend,
            )

            retrieved_ids = [
                result.chunk_id
                for result in retrieved
            ]

            relevant = case.relevant_chunk_ids
            hits = set(retrieved_ids) & relevant

            recall = len(hits) / len(relevant) if relevant else 0.0
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0

            reciprocal_rank = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            results.append(
                EvaluationResult(
                    query=case.query,
                    backend=case.backend,
                    retrieved_chunk_ids=retrieved_ids,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    reciprocal_rank=reciprocal_rank,
                    ndcg_at_k=_ndcg(retrieved_ids, relevant),
                )
            )

        return results