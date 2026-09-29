from typing import Literal

from pydantic import BaseModel


class EvaluationCase(BaseModel):
    query: str
    relevant_chunk_ids: set[str]
    backend: Literal["bm25", "dense", "hybrid"] = "hybrid"


class EvaluationResult(BaseModel):
    query: str
    backend: Literal["bm25", "dense", "hybrid"]
    retrieved_chunk_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class BenchmarkResult(BaseModel):
    backend: Literal["bm25", "dense", "hybrid"]
    query_count: int
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    average_latency_ms: float
    p95_latency_ms: float