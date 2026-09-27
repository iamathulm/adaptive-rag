from app.core.config import settings
from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import RRFFusion


class HybridRetriever:
    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.dense = DenseRetriever()
        indexed_documents = (
            documents
            if documents is not None
            else self.dense.vector_store.get_documents()
        )
        self.bm25 = BM25Retriever(indexed_documents)
        self.fusion = RRFFusion()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        limit = settings.retrieval_top_k if top_k is None else top_k

        bm25_results = self.bm25.search(query, limit)
        dense_results = self.dense.search(query, limit)

        ranked_lists = [
            bm25_results,
            dense_results,
        ]

        fused = self.fusion.fuse(ranked_lists, limit)

        return [
            RetrievalResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=score,
                source="hybrid",
            )
            for result, score in fused
        ]