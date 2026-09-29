from app.models.retrieval import RetrievalResult
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore


class DenseRetriever:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        vector = self.embedder.embed(query)
        results = self.vector_store.search(
            vector,
            top_k=top_k,
            document_id=document_id,
            source=source,
        )

        return [
            RetrievalResult(
                document_id=str(result.payload.get("document_id", result.id)),
                chunk_id=str(result.payload.get("chunk_id", result.id)),
                content=str(result.payload.get("content", "")),
                score=float(result.score),
                source=str(result.payload.get("source", "dense")),
                page=result.payload.get("page"),
                content_hash=result.payload.get("content_hash"),
            )
            for result in results
        ]