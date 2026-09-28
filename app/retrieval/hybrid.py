from typing import Literal

from app.core.config import settings
from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import RRFFusion


class HybridRetriever:
    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever(documents or [])
        self._documents_loaded = documents is not None
        self.fusion = RRFFusion()

    def refresh(self) -> None:
        self.bm25 = BM25Retriever(self.dense.vector_store.get_documents())
        self._documents_loaded = True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        backend: Literal["bm25", "dense", "hybrid"] = "hybrid",
        document_id: str | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        limit = settings.retrieval_top_k if top_k is None else top_k

        if not self._documents_loaded:
            self.refresh()

        if backend == "bm25":
            return self.bm25.search(
                query,
                limit,
                document_id=document_id,
                source=source,
            )[:limit]
        if backend == "dense":
            return self.dense.search(
                query,
                limit,
                document_id=document_id,
                source=source,
            )[:limit]

        bm25_results = self.bm25.search(
            query,
            limit,
            document_id=document_id,
            source=source,
        )
        dense_results = self.dense.search(
            query,
            limit,
            document_id=document_id,
            source=source,
        )

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