from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, documents: list[str]) -> None:
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        dense_results = self.dense.search(query, top_k)
        return dense_results