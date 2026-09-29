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


def _retrieval_kwargs(request) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if request.backend != "hybrid":
        kwargs["backend"] = request.backend
    if request.document_id is not None:
        kwargs["document_id"] = request.document_id
    if request.source is not None:
        kwargs["source"] = request.source
    return kwargs

@router.post("", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> RetrievalResponse:
    results = await run_in_threadpool(
        retriever.search,
        request.query,
        request.top_k,
        **_retrieval_kwargs(request),
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


@answer_router.post(
    "",
    response_model=AnswerResponse,
    response_model_exclude_none=True,
)
async def answer(
    request: AnswerRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    generator: Annotated[AnswerGenerator, Depends(get_generator)],
) -> AnswerResponse:
    results = await run_in_threadpool(
        retriever.search,
        request.query,
        request.top_k,
        **_retrieval_kwargs(request),
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