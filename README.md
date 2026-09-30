# Adaptive Multimodal RAG

Adaptive Multimodal RAG is a local-first retrieval-augmented generation system built with FastAPI, Qdrant, BM25, hybrid retrieval, and a Gemini-first generation layer. It is designed to stay practical for a zero-cost capstone workflow while still being production-minded in structure, observability, and failure handling.

## Highlights

- Retrieval with BM25, dense embeddings, and hybrid fusion
- Qdrant-backed storage for vector search and indexing
- Document ingestion pipeline with chunking and provenance metadata
- Provider abstraction for mock and Gemini generation
- Input validation, safe failure handling, and request logging
- Free-tier protection for Gemini through concurrency and quota guards
- Dockerized deployment for local or small-scale production-like runs

## Architecture

- API layer: FastAPI app with retrieval, answer, ingest, and health endpoints
- Retrieval layer: BM25 + dense + hybrid ranking logic with Qdrant support
- Ingestion layer: file conversion, chunking, indexing, and metadata preservation
- Generation layer: provider-neutral generator with Gemini and mock implementations
- Config and safety: environment-based settings and rate-limit guards for cost control

## Stack

- Python 3.12
- FastAPI
- Qdrant
- SentenceTransformers
- Rank-BM25
- Docling
- Gemini API

## Quick start

### 1) Create the environment file

Copy the sample configuration and add your own values:

```bash
cp .env.example .env
```

For a local Gemini-only setup, make sure the file contains:

```env
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit the real `.env` file to source control.

### 2) Install dependencies

```bash
uv sync --dev
```

### 3) Start Qdrant

```bash
docker compose up -d qdrant
```

### 4) Run the API locally

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) Confirm health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

## Docker deployment

```bash
docker compose up --build
```

This starts the API and Qdrant together. The service reads its runtime config from `.env` and the Qdrant container stores persistent data in a named Docker volume.

## API endpoints

- `GET /health` and `GET /health/ready`
- `GET /retrieval` and `POST /retrieval`
- `POST /answer`
- `POST /ingest`

## Gemini free-tier guidance

The project is intentionally conservative when using Gemini free-tier access:

- default request caps are set to a low threshold
- generation concurrency is limited to one in-flight request
- the app refuses or throttles requests when rate limits are approached
- the app supports a mock generation mode for local testing without API usage

To keep usage low in practice:

- prefer `GENERATION_PROVIDER=mock` for development and demos unless live generation is required
- keep `GENERATION_REQUESTS_PER_MINUTE` and `GENERATION_REQUESTS_PER_DAY` low
- avoid high-throughput benchmarking without a paid quota or explicit testing plan

## Project layout

```text
app/
  api/
  core/
  evaluation/
  generation/
  ingestion/
  models/
  providers/
  reranking/
  retrieval/
scripts/
tests/
Dockerfile
docker-compose.yml
pyproject.toml
README.md
```

## Verification

The project is validated with:

```bash
uv run pytest -q
uv run ruff check .
```

These checks are expected to pass before deployment or release.

## Notes

This repository is designed for a capstone or demo environment rather than large enterprise-scale production. The code is intentionally structured to support reliable local-first usage, controlled AI spend, and clean operational readiness without hardcoding secrets into the codebase.
