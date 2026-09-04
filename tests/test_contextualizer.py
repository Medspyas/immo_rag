import pytest

from app.config import Settings
from app.rag.chunking import RawChunk
from app.rag.contextualizer import (
    ContextGenerator,
    GroqContextGenerator,
    contextualiser_chunks,
    get_context_generator,
)


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x", "groq_api_key": "fake-key"}
    base.update(overrides)
    return Settings(**base)


class _FakeGenerator(ContextGenerator):
    """Générateur factice pour tester contextualiser_chunks sans appeler un LLM réel."""

    def generer_contexte(self, document: str, chunk: str) -> str:
        return "Contexte factice."


def test_groq_est_le_fournisseur_par_defaut():
    provider = get_context_generator(_settings())
    assert isinstance(provider, GroqContextGenerator)


def test_groq_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_context_generator(_settings(groq_api_key=None))


def test_anthropic_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_context_generator(_settings(llm_provider="anthropic", anthropic_api_key=None))


def test_fournisseur_inconnu_leve_une_erreur():
    with pytest.raises(ValueError):
        get_context_generator(_settings(llm_provider="mistral"))


def test_contextualiser_prefixe_le_contenu_original():
    chunk = RawChunk(doc_id="loi-1965", article="1", content="Texte de l'article.")

    resultat = contextualiser_chunks([chunk], document_complet="Document entier.", generator=_FakeGenerator())

    assert len(resultat) == 1
    assert resultat[0].content == "Contexte factice.\n\nTexte de l'article."


def test_contextualiser_ne_modifie_pas_les_chunks_d_origine():
    chunk = RawChunk(doc_id="loi-1965", article="1", content="Texte original.")

    contextualiser_chunks([chunk], document_complet="Document entier.", generator=_FakeGenerator())

    assert chunk.content == "Texte original."


def test_contextualiser_conserve_doc_id_article_et_abroge():
    chunk = RawChunk(doc_id="loi-1965", article="6-1", content="Texte.", abroge=True)

    resultat = contextualiser_chunks([chunk], document_complet="Doc.", generator=_FakeGenerator())

    assert resultat[0].doc_id == "loi-1965"
    assert resultat[0].article == "6-1"
    assert resultat[0].abroge is True
