from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from routers.analyze import router as analyze_router
from routers.health import router as health_router
from routers.ml import router as ml_router
from routers.rag import router as rag_router
from routers.retrieve import router as retrieve_router
from services.logging_service import get_logger, setup_logging


load_dotenv()
setup_logging()
logger = get_logger("decision_intelligence.api")

default_origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(title="Semantic Retrieval API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins + frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analyze_router)
app.include_router(health_router)
app.include_router(ml_router)
app.include_router(retrieve_router)
app.include_router(rag_router)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    import time

    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "request_failed method=%s path=%s latency_ms=%s",
            request.method,
            request.url.path,
            latency_ms,
        )
        raise

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "request_completed method=%s path=%s status_code=%s latency_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response
