"""Configuration du projet, lue depuis .env (cf. .env.example).

extra="forbid" : une clé en trop ou mal orthographiée dans .env (ex.
VOYAGE_API_KEYY) fait planter le démarrage avec une erreur claire, plutôt que
d'être silencieusement ignorée et de laisser tourner l'app avec une clé vide.
"""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    database_url: str

    # Literal plutôt que str : un typo dans .env (ex. "voyag") est rejeté au
    # démarrage au lieu d'échouer plus tard dans get_embedding_provider avec
    # un message moins clair.
    embeddings_provider: Literal["local", "voyage", "openai"] = "local"
    embeddings_model: str = "intfloat/multilingual-e5-base"
    embeddings_dimension: int = 768

    voyage_api_key: str | None = None
    openai_api_key: str | None = None

    llm_provider: Literal["groq", "anthropic"] = "groq"
    llm_model: str = "openai/gpt-oss-20b"

    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
