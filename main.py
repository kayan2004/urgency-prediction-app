from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.analyze import router as analyze_router
from routers.health import router as health_router
from routers.ml import router as ml_router
from routers.rag import router as rag_router
from routers.retrieve import router as retrieve_router


load_dotenv()

default_origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
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
