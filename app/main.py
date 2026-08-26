"""
Point d'entrée FastAPI.

Les briques RAG (embeddings, retrieval, génération) vivent dans app/rag/ et
sont testées indépendamment de la couche HTTP. Ce fichier ne fait qu'exposer
ce qui existe déjà, au fur et à mesure que les briques du "cœur IA" sont
prêtes (cf. CLAUDE.md, ordre de construction) — /ingest et /query arrivent
avec pipeline.py, une fois contextualizer/reranking/generator branchés.
"""
from __future__ import annotations

import psycopg
from fastapi import FastAPI

from app.config import Settings
from app.models import HealthResponse

app = FastAPI(title="immo-rag")
settings = Settings()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        psycopg.connect(settings.database_url, connect_timeout=2).close()
        db_status = "ok"
    except psycopg.OperationalError:
        db_status = "unreachable"
    return HealthResponse(status="ok", db=db_status)
