from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import RRFFusion


class HybridRetriever:
    def __init__(self, documents: list[str]) -> None:
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever()
        self.fusion = RRFFusion()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        bm25_results = self.bm25.search(query, top_k)
        dense_results = self.dense.search(query, top_k)

        ranked_lists = [
            [document for document, _ in bm25_results],
            [result.content for result in dense_results],
        ]

        fused = self.fusion.fuse(ranked_lists, top_k)

        dense_by_content = {
            result.content: result
            for result in dense_results
        }

        return [
            RetrievalResult(
                document_id=dense_by_content[content].document_id,
                content=content,
                score=score,
                source="hybrid",
            )
            for content, score in fused
            if content in dense_by_content
        ]