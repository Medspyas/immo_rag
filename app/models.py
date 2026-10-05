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

QUESTION_MAX_CARACTERES = 1000


class HealthResponse(BaseModel):
    status: str
    db: str


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    texte: str = Field(min_length=1)
    contextualiser: bool = False


class QueryRequest(BaseModel):
    # Plafond de longueur : une question métier tient en quelques phrases, et
    # un texte démesuré consommerait le quota du LLM (et du reclasseur) pour
    # rien.
    question: str = Field(min_length=1, max_length=QUESTION_MAX_CARACTERES)
