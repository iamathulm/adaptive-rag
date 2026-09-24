class RRFFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        ranked_lists: list[list[str]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, document in enumerate(ranked_list, start=1):
                scores[document] = scores.get(document, 0.0) + 1 / (
                    self.k + rank
                )

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]