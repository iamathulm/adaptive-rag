from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as retrieval_router
from app.core.config import settings
from app.core.logging import configure_logging, logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger.info("application_started", environment=settings.environment)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(retrieval_router)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}