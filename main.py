from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from routers.analyze import router as analyze_router
from routers.health import router as health_router
from routers.ml import router as ml_router
from routers.rag import router as rag_router
from routers.retrieve import router as retrieve_router


load_dotenv()

app = FastAPI(title="Semantic Retrieval API")
app.include_router(analyze_router)
app.include_router(health_router)
app.include_router(ml_router)
app.include_router(retrieve_router)
app.include_router(rag_router)
