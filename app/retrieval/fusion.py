from app.core.config import settings
from app.models.retrieval import RetrievalResult


class RRFFusion:
    def __init__(self, k: int | None = None) -> None:
        self.k = settings.rrf_k if k is None else k

    def fuse(
        self,
        ranked_lists: list[list[RetrievalResult]],
        top_k: int = 5,
    ) -> list[tuple[RetrievalResult, float]]:
        scores: dict[str, float] = {}
        documents: dict[str, RetrievalResult] = {}

        for ranked_list in ranked_lists:
            for rank, document in enumerate(ranked_list, start=1):
                chunk_id = document.chunk_id
                documents.setdefault(chunk_id, document)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (
                    self.k + rank
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            (documents[chunk_id], score)
            for chunk_id, score in ranked
        ]