from pydantic import BaseModel


class EvaluationCase(BaseModel):
    query: str
    relevant_chunk_ids: set[str]


class EvaluationResult(BaseModel):
    query: str
    retrieved_chunk_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float