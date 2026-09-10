"""Chaîne d'ingestion : d'un texte de loi aux morceaux prêts à rechercher.

Les briques précédentes traitent chacune une étape isolée (découpage, contexte,
embeddings, rangement). Ce module les met bout à bout : un texte de loi entre,
des morceaux indexés sortent.

Étapes enchaînées (cf. CLAUDE.md, section Flux) :
  1. découper le texte en articles            (chunking.py)
  2. situer chaque article dans la loi         (contextualizer.py, facultatif)
  3. transformer chaque article en vecteur     (embeddings.py)
  4. ranger les morceaux dans la base           (vector_store.py)

L'étape 2 est facultative : elle demande un appel de modèle de langage par
article. Sans elle, le texte brut de l'article est indexé tel quel.

La recherche par mots-clés (BM25) n'est pas construite ici : son index se bâtit
d'un bloc à partir de tous les morceaux d'un coup (bm25_index.py). documents_bm25
en fournit la matière à partir des morceaux préparés.

Contenu du fichier :
- RapportIngestion : bilan chiffré d'une ingestion (combien d'articles, dont
  combien d'abrogés).
- identifiant_chunk : identifiant stable d'un article — même loi et même numéro
  donnent toujours le même identifiant, d'une ingestion à l'autre.
- preparer_morceaux : étapes 1 à 3, sans toucher à la base.
- ingerer_document : les quatre étapes, rangement en base compris.
- documents_bm25 : convertit des morceaux préparés en entrées pour l'index BM25.
"""
from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel

from app.rag.bm25_index import BM25Document
from app.rag.chunking import decouper_en_articles
from app.rag.contextualizer import ContextGenerator, contextualiser_chunks
from app.rag.vector_store import Chunk

# Espace de noms fixe pour dériver l'identifiant d'un article de façon
# déterministe. Conséquence : réingérer la même loi réécrit les mêmes morceaux
# (upsert) au lieu d'en créer des doublons.
NS_CHUNK = UUID("00000000-0000-0000-0000-000000000001")


class _Embeddings(Protocol):
    """Partie du fournisseur d'embeddings utilisée ici : la vectorisation de
    plusieurs textes à ranger."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class _VectorStore(Protocol):
    """Partie de la base vectorielle utilisée ici : remplacer les morceaux d'un
    document, puis insérer les nouveaux."""

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def delete_by_doc(self, doc_id: str) -> None: ...


class RapportIngestion(BaseModel):
    """Bilan d'une ingestion, pour vérifier d'un coup d'œil qu'une loi est
    entrée correctement."""

    doc_id: str
    articles_indexes: int
    articles_abroges: int
    contextualise: bool


def identifiant_chunk(doc_id: str, article: str) -> UUID:
    """Identifiant stable d'un article, dérivé de sa loi et de son numéro.

    Deux ingestions du même texte produisent les mêmes identifiants : la seconde
    remplace la première au lieu d'ajouter des doublons. Les deux recherches
    (sens et mots-clés) partagent ainsi le même identifiant pour un article
    donné, ce dont la fusion RRF a besoin.
    """
    return uuid5(NS_CHUNK, f"{doc_id}:{article}")


def preparer_morceaux(
    texte: str,
    doc_id: str,
    *,
    embeddings: _Embeddings,
    contextualizer: ContextGenerator | None = None,
) -> list[Chunk]:
    """Découpe, situe (facultatif) puis vectorise un texte de loi.

    Renvoie les morceaux prêts à ranger, sans toucher à la base. Utile quand on
    veut garder les morceaux en mémoire (évaluation, démo) plutôt que les
    envoyer dans PostgreSQL.

    doc_id nomme la loi (en général le nom du fichier sans extension) ; il relie
    entre eux tous les morceaux d'un même texte.

    Lève une erreur si le texte ne contient aucun article exploitable.
    """
    articles = decouper_en_articles(texte, doc_id)
    if not articles:
        raise ValueError(
            f"Aucun article trouvé dans le texte de « {doc_id} » : rien à indexer."
        )

    if contextualizer is not None:
        articles = contextualiser_chunks(articles, texte, contextualizer)

    vecteurs = embeddings.embed_documents([a.content for a in articles])

    return [
        Chunk(
            id=identifiant_chunk(doc_id, article.article),
            doc_id=doc_id,
            content=article.content,
            embedding=vecteur,
            metadata={"article": article.article, "abroge": article.abroge},
        )
        for article, vecteur in zip(articles, vecteurs, strict=True)
    ]


def ingerer_document(
    texte: str,
    doc_id: str,
    *,
    embeddings: _Embeddings,
    vector_store: _VectorStore,
    contextualizer: ContextGenerator | None = None,
) -> RapportIngestion:
    """Enchaîne les quatre étapes, rangement en base compris.

    Les anciens morceaux de la loi sont retirés avant l'insertion des nouveaux :
    sans cela, un article supprimé d'une version à jour resterait en base.
    """
    morceaux = preparer_morceaux(
        texte, doc_id, embeddings=embeddings, contextualizer=contextualizer
    )

    vector_store.delete_by_doc(doc_id)
    vector_store.upsert(morceaux)

    return RapportIngestion(
        doc_id=doc_id,
        articles_indexes=len(morceaux),
        articles_abroges=sum(1 for m in morceaux if m.metadata.get("abroge")),
        contextualise=contextualizer is not None,
    )


def documents_bm25(morceaux: list[Chunk]) -> list[BM25Document]:
    """Convertit des morceaux préparés en entrées pour l'index BM25, qui n'a pas
    besoin des vecteurs."""
    return [
        BM25Document(
            id=m.id, doc_id=m.doc_id, content=m.content, metadata=m.metadata
        )
        for m in morceaux
    ]
