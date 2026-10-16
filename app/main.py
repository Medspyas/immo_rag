"""Point d'entrée de l'API web (FastAPI).

Routes : /health (le serveur répond, la base est-elle joignable ?), /ingest
(indexer un texte de loi), /query (poser une question), et /documents (lister
les textes indexés, en retirer un). La logique
de recherche et de génération réside dans app/rag/ et se teste indépendamment
de l'API ; ce module ne fait qu'assembler les composants à partir de .env et
les brancher sur les routes.

Les composants qui chargent un modèle ou ouvrent une connexion (embeddings,
reclasseur, base vectorielle, index BM25...) sont construits paresseusement,
au premier appel, et réutilisés ensuite (_cache) : la construction échoue si
la base ou un modèle est indisponible, et il ne faut pas empêcher /health de
répondre pour autant. Un échec de construction devient une réponse 503 plutôt
qu'un plantage du serveur.

FastAPI exécute les routes synchrones dans plusieurs threads : un verrou
garantit que deux premières requêtes simultanées ne chargent pas chacune les
modèles (plusieurs Go de mémoire en double).
"""
from __future__ import annotations

import threading

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.models import HealthResponse, IngestRequest, QueryRequest
from app.rag.contextualizer import ContextGenerator, get_context_generator
from app.rag.embeddings import get_embedding_provider
from app.rag.generator import (
    LLM,
    ModeleIndisponible,
    ReponseGeneree,
    get_llm,
    get_llm_juge,
)
from app.rag.nli import NLIProvider, get_nli_provider
from app.rag.pipeline import (
    RapportIngestion,
    construire_index_bm25,
    construire_retriever,
    ingerer_document,
    repondre,
)
from app.rag.reranker import get_rerank_provider
from app.rag.vector_store import DocumentIndexe, VectorStore

app = FastAPI(title="immo-rag")
settings = Settings()

# Le front (phase 3) est servi depuis une autre origine que l'API : le
# navigateur bloque ses appels sauf origine autorisée ici. Liste vide par
# défaut : aucune origine extérieure acceptée.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

# Composants construits paresseusement, une seule fois (cf. docstring). Une
# ingestion réussie invalide "bm25" et "retriever" : le corpus a changé, l'un
# et l'autre doivent être reconstruits à la prochaine question.
_cache: dict[str, object] = {}
# Réentrant : _retriever appelle _get pendant qu'il tient déjà le verrou.
_verrou = threading.RLock()


def _get(nom: str, construire):
    with _verrou:
        if nom not in _cache:
            _cache[nom] = construire()
        return _cache[nom]


def _invalider_recherche() -> None:
    """Le corpus a changé : l'index BM25 en cache et le retriever qui l'utilise
    sont périmés, reconstruits à la prochaine question."""
    with _verrou:
        _cache.pop("bm25", None)
        _cache.pop("retriever", None)


def _vector_store() -> VectorStore:
    return _get(
        "vector_store",
        lambda: VectorStore(settings.database_url, settings.embeddings_dimension),
    )


def _retriever():
    with _verrou:
        if "retriever" not in _cache:
            embeddings = _get("embeddings", lambda: get_embedding_provider(settings))
            reranker = _get("reranker", lambda: get_rerank_provider(settings))
            bm25 = _get("bm25", lambda: construire_index_bm25(_vector_store()))
            _cache["retriever"] = construire_retriever(
                settings, embeddings, _vector_store(), bm25, reranker
            )
        return _cache["retriever"]


def _llm() -> LLM:
    return _get("llm", lambda: get_llm(settings))


def _juge() -> LLM:
    return _get("juge", lambda: get_llm_juge(settings))


def _nli() -> NLIProvider:
    return _get("nli", lambda: get_nli_provider(settings))


def _contextualizer() -> ContextGenerator:
    return get_context_generator(settings)


# Erreurs attendues lorsqu'un composant est indisponible : base injoignable,
# réglage invalide (ex. clé API manquante). Toute autre erreur remonte telle
# quelle, comme une erreur 500 non anticipée.
_ERREURS_COMPOSANT_INDISPONIBLE = (psycopg.OperationalError, ValueError)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        psycopg.connect(settings.database_url, connect_timeout=2).close()
        db_status = "ok"
    except psycopg.OperationalError:
        db_status = "unreachable"
    return HealthResponse(status="ok", db=db_status)


@app.post("/ingest", response_model=RapportIngestion)
def ingest(payload: IngestRequest) -> RapportIngestion:
    try:
        embeddings = _get("embeddings", lambda: get_embedding_provider(settings))
        contextualizer = _contextualizer() if payload.contextualiser else None
        rapport = ingerer_document(
            payload.texte,
            payload.doc_id,
            embeddings=embeddings,
            vector_store=_vector_store(),
            contextualizer=contextualizer,
        )
    except _ERREURS_COMPOSANT_INDISPONIBLE as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _invalider_recherche()
    return rapport


@app.get("/documents", response_model=list[DocumentIndexe])
def documents() -> list[DocumentIndexe]:
    try:
        return _vector_store().documents()
    except _ERREURS_COMPOSANT_INDISPONIBLE as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/documents/{doc_id}", status_code=204)
def supprimer_document(doc_id: str) -> None:
    try:
        supprimes = _vector_store().delete_by_doc(doc_id)
    except _ERREURS_COMPOSANT_INDISPONIBLE as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not supprimes:
        raise HTTPException(status_code=404, detail=f"document inconnu : {doc_id}")
    _invalider_recherche()


@app.post("/query", response_model=ReponseGeneree)
def query(payload: QueryRequest) -> ReponseGeneree:
    try:
        retriever = _retriever()
        llm = _llm()
        juge = _juge()
        nli = _nli()
    except _ERREURS_COMPOSANT_INDISPONIBLE as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return repondre(
            payload.question, retriever, llm, nli, settings, juge=juge,
            historique=payload.historique,
        )
    except ModeleIndisponible as exc:
        # Tous les modèles saturés, même après attente : service momentanément
        # indisponible, pas une erreur du serveur.
        raise HTTPException(
            status_code=503, detail="modèle de langage saturé, réessayer plus tard"
        ) from exc
