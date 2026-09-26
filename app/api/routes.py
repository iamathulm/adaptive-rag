from fastapi import APIRouter, Depends

from app.api.dependencies import get_dense_retriever
from app.models.api import RetrievalRequest, RetrievalResponse
from app.retrieval.dense import DenseRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
retriever_dependency = Depends(get_dense_retriever)

@router.post("", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    retriever: DenseRetriever = retriever_dependency,
) -> RetrievalResponse:
    results = retriever.search(
        request.query,
        top_k=request.top_k,
    )
    return RetrievalResponse(results=results)