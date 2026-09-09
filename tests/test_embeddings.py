import pytest

from app.config import Settings
from app.rag.embeddings import LocalEmbeddings, get_embedding_provider


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x"}
    base.update(overrides)
    return Settings(**base)


def test_local_ne_necessite_aucune_cle_api():
    settings = _settings(embeddings_provider="local")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, LocalEmbeddings)
    assert provider.dimension == settings.embeddings_dimension


def test_local_ne_charge_pas_le_modele_a_la_construction():
    provider = get_embedding_provider(_settings())
    assert provider._model is None


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(embeddings_provider="voyage")
