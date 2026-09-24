from pydantic import BaseModel


class RetrievalResult(BaseModel):
    document_id: str
    content: str
    score: float
    source: str