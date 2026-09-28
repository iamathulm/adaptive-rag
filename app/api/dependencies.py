from functools import lru_cache

from app.core.config import settings
from app.generation.generator import AnswerGenerator, create_generator
from app.retrieval.hybrid import HybridRetriever


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


@lru_cache
def get_generator() -> AnswerGenerator:
    return create_generator(settings)