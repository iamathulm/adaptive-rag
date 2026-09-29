import json
import math
import time
from pathlib import Path

from app.models.evaluation import BenchmarkResult, EvaluationCase, EvaluationResult
from app.retrieval.hybrid import HybridRetriever


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(case) for case in data]


def benchmark_retrieval(
    retriever: HybridRetriever,
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> list[BenchmarkResult]:
    benchmark_results: list[BenchmarkResult] = []

    for backend in ("bm25", "dense", "hybrid"):
        evaluator = RetrievalEvaluator(retriever)
        case_results: list[EvaluationResult] = []
        latencies: list[float] = []

        for case in cases:
            backend_case = case.model_copy(update={"backend": backend})
            started = time.perf_counter()
            case_results.extend(evaluator.evaluate([backend_case], top_k=top_k))
            latencies.append((time.perf_counter() - started) * 1000)

        query_count = len(case_results)
        if query_count == 0:
            benchmark_results.append(
                BenchmarkResult(
                    backend=backend,
                    query_count=0,
                    recall_at_k=0.0,
                    precision_at_k=0.0,
                    reciprocal_rank=0.0,
                    ndcg_at_k=0.0,
                    average_latency_ms=0.0,
                    p95_latency_ms=0.0,
                )
            )
            continue

        ordered_latencies = sorted(latencies)
        p95_index = min(math.ceil(query_count * 0.95) - 1, query_count - 1)
        benchmark_results.append(
            BenchmarkResult(
                backend=backend,
                query_count=query_count,
                recall_at_k=sum(result.recall_at_k for result in case_results) / query_count,
                precision_at_k=sum(result.precision_at_k for result in case_results)
                / query_count,
                reciprocal_rank=sum(result.reciprocal_rank for result in case_results)
                / query_count,
                ndcg_at_k=sum(result.ndcg_at_k for result in case_results) / query_count,
                average_latency_ms=sum(latencies) / query_count,
                p95_latency_ms=ordered_latencies[p95_index],
            )
        )

    return benchmark_results


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