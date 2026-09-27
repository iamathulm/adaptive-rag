from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_retriever
from app.models.api import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseItem,
)
from app.retrieval.hybrid import HybridRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

@router.post("", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> RetrievalResponse:
    results = await run_in_threadpool(
        retriever.search,
        request.query,
        request.top_k,
    )
    return RetrievalResponse(
        results=[
            RetrievalResponseItem(
                document_id=result.document_id,
                content=result.content,
                score=result.score,
                source=result.source,
            )
            for result in results
        ]
    )