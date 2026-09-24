from app.models.retrieval import RetrievalResult
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore


class DenseRetriever:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        vector = self.embedder.embed(query)

        results = self.vector_store.client.query_points(
            collection_name="documents",
            query=vector,
            limit=top_k,
        ).points

        return [
            RetrievalResult(
                document_id=str(result.id),
                content=str(result.payload.get("content", "")),
                score=float(result.score),
                source="dense",
            )
            for result in results
        ]