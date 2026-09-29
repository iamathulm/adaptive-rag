from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import answer_router, ingest_router
from app.api.routes import router as retrieval_router
from app.core.config import settings
from app.core.logging import configure_logging, logger
from app.retrieval import VectorStore


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()

    vector_store = VectorStore()
    vector_store.ensure_collection()

    logger.info(
        "application_started",
        environment=settings.environment,
    )

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(retrieval_router)
app.include_router(answer_router)
app.include_router(ingest_router)


@app.get("/health/live")
async def liveness_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return await liveness_check()


@app.get("/health/ready")
async def readiness_check() -> JSONResponse:
    checks: dict[str, str] = {
        "qdrant": "ok",
        "generation": "ok",
    }

    try:
        VectorStore().client.get_collections()
    except Exception:
        checks["qdrant"] = "unavailable"

    if settings.generation_provider == "gemini" and settings.gemini_api_key is None:
        checks["generation"] = "not_configured"

    is_ready = all(status == "ok" for status in checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ok" if is_ready else "unavailable", "checks": checks},
    )