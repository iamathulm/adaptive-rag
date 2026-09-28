from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_generator, get_retriever
from app.generation.citations import validate_citations
from app.generation.generator import AnswerGenerator, GenerationError
from app.models.answer import AnswerRequest, AnswerResponse
from app.models.api import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseItem,
)
from app.retrieval.hybrid import HybridRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
answer_router = APIRouter(prefix="/answer", tags=["answer"])

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


@answer_router.post("", response_model=AnswerResponse)
async def answer(
    request: AnswerRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    generator: Annotated[AnswerGenerator, Depends(get_generator)],
) -> AnswerResponse:
    results = await run_in_threadpool(
        retriever.search,
        request.query,
        request.top_k,
    )

    try:
        generated_answer = await run_in_threadpool(
            generator.generate,
            request.query,
            results,
        )
    except GenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "generation_failed",
                "message": "The answer provider could not generate a response",
            },
        ) from exc

    citations, grounded = validate_citations(generated_answer, results)
    return AnswerResponse(
        answer=generated_answer,
        citations=citations,
        grounded=grounded,
        sources=results,
    )