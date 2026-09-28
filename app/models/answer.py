from pydantic import BaseModel

from app.models.retrieval import RetrievalResult


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    answer: str
    sources: list[RetrievalResult]