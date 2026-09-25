import pytest

from app.config import Settings
from app.rag.chunking import RawChunk
from app.rag.contextualizer import (
    LONGUEUR_MAX_RESUME_CARACTERES,
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
    """Générateur de substitution : aucun appel à un vrai modèle pendant les
    tests. Retient les appels reçus pour vérifier qu'un seul a lieu par
    document, quel que soit le nombre de morceaux."""

    def __init__(self, resume: str = "Contexte factice."):
        self._resume = resume
        self.debuts_recus: list[str] = []

    def resumer_document(self, debut_document: str) -> str:
        self.debuts_recus.append(debut_document)
        return self._resume


def test_groq_est_le_fournisseur_par_defaut():
    provider = get_context_generator(_settings())
    assert isinstance(provider, GroqContextGenerator)


def test_client_groq_reessaie_sur_limite_de_debit():
    from app.rag.generator import GROQ_MAX_ESSAIS

    provider = get_context_generator(_settings())
    assert provider._client.max_retries == GROQ_MAX_ESSAIS


def test_groq_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_context_generator(_settings(groq_api_key=None))


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(llm_provider="anthropic")


def test_contextualiser_prefixe_le_resume_et_le_numero_d_article():
    chunk = RawChunk(doc_id="loi-1965", article="1", content="Texte de l'article.")

    resultat = contextualiser_chunks([chunk], document_complet="Document entier.", generator=_FakeGenerator())

    assert len(resultat) == 1
    assert resultat[0].content == "Contexte factice. Article 1.\n\nTexte de l'article."


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


def test_contextualiser_un_seul_appel_au_generateur_quel_que_soit_le_nombre_de_morceaux():
    chunks = [
        RawChunk(doc_id="loi-1965", article=str(i), content=f"Texte {i}.")
        for i in range(1, 6)
    ]
    generateur = _FakeGenerator()

    resultat = contextualiser_chunks(chunks, document_complet="Document entier.", generator=generateur)

    assert len(generateur.debuts_recus) == 1
    assert all(r.content.startswith("Contexte factice. Article ") for r in resultat)


def test_contextualiser_tronque_le_document_transmis_au_generateur():
    document_long = "x" * (LONGUEUR_MAX_RESUME_CARACTERES + 5_000)
    chunk = RawChunk(doc_id="loi-1965", article="1", content="Texte.")
    generateur = _FakeGenerator()

    contextualiser_chunks([chunk], document_complet=document_long, generator=generateur)

    assert len(generateur.debuts_recus[0]) == LONGUEUR_MAX_RESUME_CARACTERES


def test_contextualiser_liste_vide_ne_genere_aucun_appel():
    generateur = _FakeGenerator()

    resultat = contextualiser_chunks([], document_complet="Document.", generator=generateur)

    assert resultat == []
    assert generateur.debuts_recus == []
