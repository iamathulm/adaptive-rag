from functools import lru_cache

from app.retrieval.dense import DenseRetriever


@lru_cache
def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever()