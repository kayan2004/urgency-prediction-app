# Urgency Prediction Backend

FastAPI backend for support-ticket analysis, retrieval-augmented answering, and priority prediction.

## Features

- Semantic retrieval over support tickets
- RAG and non-RAG answer generation
- ML priority prediction
- LLM zero-shot priority prediction
- Aggregated `/analyze` endpoint for end-to-end comparison

## Main Endpoints

- `GET /health`
- `POST /retrieve`
- `POST /rag-answer`
- `POST /non-rag-answer`
- `POST /priority-predict`
- `POST /priority-predict-llm`
- `POST /analyze`

## Project Structure

- `routers/`: FastAPI route definitions
- `schemas/`: Pydantic request and response models
- `services/`: retrieval, generation, orchestration, and ML logic
- `model/`: runtime model artifacts
- `dataset/retrieval/`: retrieval dataset used by the API

## Required Runtime Files

The deployed backend expects these files to exist:

- `model/best_logistic_regression.joblib`
- `model/rag_voyage_embeddings.npy`
- `dataset/retrieval/support_tickets_for_rag.csv`

## Environment Variables

- `GEMINI_API_KEY`
- `VOYAGE_API_KEY`
- `FRONTEND_ORIGINS`
- optional: `GEMINI_TEXT_MODEL`

## Local Development

Install dependencies:

```bash
uv sync
```

Run the API locally:

```bash
uv run uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build:

```bash
docker build -t urgency-backend .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 urgency-backend
```

The container binds to `${PORT}` automatically when deployed on platforms like Railway.

## Deployment Notes

- The backend Docker image is trimmed with `.dockerignore` to avoid shipping raw datasets and old large artifacts.
- CORS allows local development origins by default and accepts deployment origins through `FRONTEND_ORIGINS`.
