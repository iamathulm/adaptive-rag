import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_generator, get_ingestion_service, get_retriever
from app.core.config import settings
from app.generation.citations import validate_citations
from app.generation.generator import AnswerGenerator, GenerationError
from app.generation.limiter import GenerationRateLimitError
from app.ingestion import IngestionService
from app.models.answer import AnswerRequest, AnswerResponse
from app.models.api import (
    IngestionResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseItem,
)
from app.retrieval.hybrid import HybridRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
answer_router = APIRouter(prefix="/answer", tags=["answer"])
ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


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
    except GenerationRateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "generation_rate_limited",
                "message": "The generation request limit has been reached",
            },
            headers={"Retry-After": str(max(1, round(exc.retry_after_seconds)))},
        ) from exc
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


@ingest_router.post("", response_model=IngestionResponse)
async def ingest(
    file: Annotated[UploadFile, File(...)],
    document_id: Annotated[str | None, Form()] = None,
    service: Annotated[IngestionService, Depends(get_ingestion_service)] = None,
) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported")

    content = await file.read(settings.ingestion_max_file_size_bytes + 1)
    if len(content) > settings.ingestion_max_file_size_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file is too large")

    resolved_document_id = document_id or Path(file.filename).stem
    if not resolved_document_id:
        raise HTTPException(status_code=400, detail="A document ID is required")

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name

        chunks_indexed = await run_in_threadpool(
            service.ingest,
            temporary_path,
            resolved_document_id,
            source=Path(file.filename).name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ingestion_failed",
                "message": "The document could not be ingested",
            },
        ) from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

    return IngestionResponse(
        document_id=resolved_document_id,
        chunks_indexed=chunks_indexed,
    )