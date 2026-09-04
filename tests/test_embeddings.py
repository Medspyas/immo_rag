import pytest

from app.config import Settings
from app.rag.embeddings import LocalEmbeddings, get_embedding_provider


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x", "embeddings_provider": "voyage"}
    base.update(overrides)
    return Settings(**base)


def test_local_ne_necessite_aucune_cle_api():
    settings = _settings(embeddings_provider="local")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, LocalEmbeddings)
    assert provider.dimension == settings.embeddings_dimension


def test_voyage_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_embedding_provider(_settings(voyage_api_key=None))


def test_openai_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_embedding_provider(_settings(embeddings_provider="openai", openai_api_key=None))


def test_fournisseur_inconnu_leve_une_erreur():
    with pytest.raises(ValueError):
        get_embedding_provider(_settings(embeddings_provider="mistral"))
