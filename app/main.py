from fastapi import FastAPI

app = FastAPI(title="Adaptive Multimodal RAG")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}