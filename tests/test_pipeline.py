"""Tests de la chaîne d'ingestion.

Aucune base ni aucun modèle réel : les embeddings et la base vectorielle sont
remplacés par des doublures qui se contentent d'enregistrer ce qu'on leur passe.
On vérifie l'enchaînement des étapes, pas la qualité des vecteurs.
"""
from __future__ import annotations

from app.rag.pipeline import (
    documents_bm25,
    identifiant_chunk,
    ingerer_document,
    preparer_morceaux,
)
from app.rag.vector_store import Chunk

# Deux articles, dont un abrogé (marqueurs posés par le nettoyage en amont).
TEXTE = """Loi d'essai

Article 1
Le locataire donne congé avec un préavis de trois mois.

=== ARTICLE ABROGÉ (n'est plus en vigueur — ne pas citer comme droit actuel) ===
Article 2
Ancienne règle sur le dépôt de garantie.
=== FIN ARTICLE ABROGÉ ===

Article 3
Le dépôt de garantie est restitué dans un délai d'un mois.
"""


class _EmbeddingsFake:
    """Renvoie un vecteur trivial par texte et retient les textes reçus, pour
    vérifier ce qui est réellement vectorisé."""

    def __init__(self) -> None:
        self.textes_recus: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.textes_recus = list(texts)
        return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts)]


class _VectorStoreFake:
    """Enregistre l'ordre des appels et les morceaux insérés."""

    def __init__(self) -> None:
        self.appels: list[str] = []
        self.morceaux: list[Chunk] = []

    def delete_by_doc(self, doc_id: str) -> None:
        self.appels.append(f"delete:{doc_id}")

    def upsert(self, chunks: list[Chunk]) -> None:
        self.appels.append(f"upsert:{len(chunks)}")
        self.morceaux = list(chunks)


class _ContextGeneratorFake:
    def generer_contexte(self, document: str, chunk: str) -> str:
        return "[contexte]"


def test_preparer_morceaux_un_par_article_hors_preambule():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    assert len(morceaux) == 3
    assert [m.metadata["article"] for m in morceaux] == ["1", "2", "3"]


def test_preparer_morceaux_marque_l_article_abroge():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    abroges = {m.metadata["article"]: m.metadata["abroge"] for m in morceaux}
    assert abroges == {"1": False, "2": True, "3": False}


def test_preparer_morceaux_identifiants_stables_entre_deux_appels():
    a = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    b = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    assert [m.id for m in a] == [m.id for m in b]
    assert a[0].id == identifiant_chunk("loi-essai", "1")


def test_preparer_morceaux_sans_contexte_vectorise_le_texte_brut():
    embeddings = _EmbeddingsFake()
    preparer_morceaux(TEXTE, "loi-essai", embeddings=embeddings)
    assert embeddings.textes_recus[0].startswith("Article 1")


def test_preparer_morceaux_avec_contexte_prefixe_le_texte_vectorise():
    embeddings = _EmbeddingsFake()
    morceaux = preparer_morceaux(
        TEXTE,
        "loi-essai",
        embeddings=embeddings,
        contextualizer=_ContextGeneratorFake(),
    )
    assert embeddings.textes_recus[0].startswith("[contexte]")
    assert morceaux[0].content.startswith("[contexte]")


def test_preparer_morceaux_refuse_un_texte_sans_article():
    try:
        preparer_morceaux("Un texte sans le moindre article.", "vide",
                          embeddings=_EmbeddingsFake())
    except ValueError as e:
        assert "Aucun article" in str(e)
    else:
        raise AssertionError("une erreur était attendue")


def test_ingerer_document_supprime_avant_d_inserer():
    store = _VectorStoreFake()
    ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                     vector_store=store)
    assert store.appels == ["delete:loi-essai", "upsert:3"]


def test_ingerer_document_rapport_chiffre():
    store = _VectorStoreFake()
    rapport = ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                               vector_store=store)
    assert rapport.doc_id == "loi-essai"
    assert rapport.articles_indexes == 3
    assert rapport.articles_abroges == 1
    assert rapport.contextualise is False


def test_ingerer_document_signale_la_contextualisation():
    store = _VectorStoreFake()
    rapport = ingerer_document(TEXTE, "loi-essai", embeddings=_EmbeddingsFake(),
                               vector_store=store,
                               contextualizer=_ContextGeneratorFake())
    assert rapport.contextualise is True


def test_documents_bm25_reprend_id_contenu_et_metadonnees():
    morceaux = preparer_morceaux(TEXTE, "loi-essai", embeddings=_EmbeddingsFake())
    docs = documents_bm25(morceaux)
    assert [d.id for d in docs] == [m.id for m in morceaux]
    assert docs[0].metadata["article"] == "1"
    assert "préavis" in docs[0].content
