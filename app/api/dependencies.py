from functools import lru_cache

from app.retrieval.hybrid import HybridRetriever


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()