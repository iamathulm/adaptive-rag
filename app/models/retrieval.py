from pydantic import BaseModel


class RetrievalResult(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    source: str
    page: int | None = None
    content_hash: str | None = None