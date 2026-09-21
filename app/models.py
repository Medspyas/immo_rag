"""Forme des données échangées avec l'API.

Chaque classe décrit ce qu'une route reçoit ou renvoie. Les réponses des
routes /ingest et /query (RapportIngestion, ReponseGeneree) sont définies dans
app/rag/ : elles y sont produites, et une seule définition évite un doublon à
tenir synchronisé.

Contenu du fichier :
- HealthResponse : la réponse de /health.
- IngestRequest : ce que /ingest reçoit — un texte de loi à indexer.
- QueryRequest : ce que /query reçoit — une question.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    db: str


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    texte: str = Field(min_length=1)
    contextualiser: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
