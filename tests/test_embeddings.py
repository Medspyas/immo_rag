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


class _ModeleFactice:
    def __init__(self):
        self.textes = []

    def encode(self, textes, normalize_embeddings):
        self.textes.extend(textes)

        class _Tableau:
            def tolist(self):
                return [[0.0] for _ in textes]

        return _Tableau()


def test_prefixes_e5_par_defaut():
    provider = get_embedding_provider(_settings())
    provider._model = _ModeleFactice()
    provider.embed_documents(["article"])
    provider.embed_query("question")
    assert provider._model.textes == ["passage: article", "query: question"]


def test_prefixes_vides_pour_un_modele_sans_convention():
    provider = get_embedding_provider(
        _settings(embeddings_prefixe_requete="", embeddings_prefixe_passage="")
    )
    provider._model = _ModeleFactice()
    provider.embed_documents(["article"])
    provider.embed_query("question")
    assert provider._model.textes == ["article", "question"]
