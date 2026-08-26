"""Schémas Pydantic exposés par l'API (contrats de /health, /ingest, /query)."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str
