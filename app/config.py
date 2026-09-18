"""Configuration du projet.

Tous les réglages sont lus depuis le fichier .env (un modèle figure dans
.env.example). La classe Settings énumère les réglages attendus et interrompt
le démarrage si l'un d'eux manque ou si une clé est mal orthographiée.

Contrainte de projet : seuls des composants gratuits sont retenus. Chaque
réglage de fournisseur n'accepte donc qu'une valeur locale ou une offre
gratuite ; aucun service payant n'est proposé.
"""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    database_url: str

    # Embeddings : modèle local et gratuit. Un autre modèle local se choisit
    # via embeddings_model (et embeddings_dimension si sa taille diffère).
    embeddings_provider: Literal["local"] = "local"
    embeddings_model: str = "intfloat/multilingual-e5-base"
    embeddings_dimension: int = 768

    # Reclassement des résultats de recherche : modèle local et gratuit. Un
    # autre modèle local se choisit via rerank_model.
    rerank_provider: Literal["local"] = "local"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Longueur maximale, en tokens, de chaque paire (question, morceau) soumise
    # au reclasseur. bge-reranker-v2-m3 accepte jusqu'à 8192 tokens et traite
    # sinon les articles longs en entier, ce qui rend une passe CPU très lente.
    # 512 borne ce coût et aligne la fenêtre du reclasseur sur celle des embeddings e5.
    rerank_max_length: int = 512

    # LLM du contextual retrieval, puis de la génération de réponse. Groq offre
    # un accès gratuit ; le volume d'usage doit rester dans ses limites.
    llm_provider: Literal["groq"] = "groq"
    llm_model: str = "openai/gpt-oss-20b"

    groq_api_key: str | None = None
