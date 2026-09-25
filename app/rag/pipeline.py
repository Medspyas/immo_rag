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
en fournit la matière à partir des morceaux préparés ; construire_index_bm25
la fournit à partir de ce qui est déjà rangé en base (utile au démarrage de
l'application, où l'on ne repart pas des textes bruts).

Le second axe du fichier est la requête (cf. CLAUDE.md, section Flux) :
repondre enchaîne la recherche hybride, la génération de la réponse citée, et
les deux garde-fous (vérification des citations puis de la fidélité).

Contenu du fichier :
- RapportIngestion : bilan chiffré d'une ingestion (combien d'articles, dont
  combien d'abrogés).
- identifiant_chunk : identifiant stable d'un article — même loi et même numéro
  donnent toujours le même identifiant, d'une ingestion à l'autre.
- preparer_morceaux : étapes 1 à 3, sans toucher à la base.
- ingerer_document : les quatre étapes, rangement en base compris.
- documents_bm25 : convertit des morceaux préparés en entrées pour l'index BM25.
- construire_index_bm25 : reconstruit l'index BM25 à partir de la base.
- construire_retriever : assemble la recherche hybride à partir de ses
  composants et du réglage du garde-fou étage 1.
- repondre : orchestration complète d'une requête (recherche + génération +
  garde-fous), tracée dans Langfuse quand configuré (observability/tracing.py).
"""
from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel

from app.config import Settings
from app.observability.tracing import trace
from app.rag.bm25_index import BM25Document, BM25Index
from app.rag.chunking import decouper_en_articles
from app.rag.contextualizer import ContextGenerator, contextualiser_chunks
from app.rag.generator import (
    ReponseGeneree,
    generer_reponse,
    verifier_citations,
    verifier_fidelite,
)
from app.rag.nli import ScoreNLI
from app.rag.retriever import HybridRetriever
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


class _VectorStoreLecture(Protocol):
    """Partie de la base vectorielle utilisée pour reconstruire l'index BM25 :
    la lecture de tous les morceaux rangés, sans leur vecteur."""

    def tous(self) -> list: ...  # list[MorceauStocke] — non importé pour rester découplé


class _LLM(Protocol):
    """Partie d'un fournisseur de modèle de langage utilisée ici : générer une
    réponse brute, et jouer le juge de repli du garde-fou étage 2."""

    def generer(self, consigne: str, question: str) -> str: ...


class _NLI(Protocol):
    """Partie d'un fournisseur NLI utilisée ici : les scores d'implication et
    de contradiction d'extraits de texte source face à une affirmation."""

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]: ...


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


class _EmbeddingsRequete(Protocol):
    """Partie d'un fournisseur d'embeddings utilisée par le retriever : la
    vectorisation d'une question (distincte de embed_documents, utilisée à
    l'ingestion — cf. embeddings.py)."""

    def embed_query(self, text: str) -> list[float]: ...


class _RechercheDense(Protocol):
    def search(self, query_embedding: list[float], k: int) -> list: ...


class _RechercheSparse(Protocol):
    def search(self, query: str, k: int) -> list: ...


class _Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]: ...


def construire_index_bm25(vector_store: _VectorStoreLecture) -> BM25Index:
    """Reconstruit l'index BM25 (en mémoire) à partir de ce qui est déjà rangé
    en base.

    Utile au démarrage de l'application : contrairement à l'ingestion, où les
    morceaux sortent de preparer_morceaux, ici on repart de la base — c'est
    elle qui fait foi sur ce qui a été ingéré.
    """
    morceaux = vector_store.tous()
    documents = [
        BM25Document(id=m.id, doc_id=m.doc_id, content=m.content, metadata=m.metadata)
        for m in morceaux
    ]
    return BM25Index(documents)


def construire_retriever(
    settings: Settings,
    embeddings: _EmbeddingsRequete,
    vector_store: _RechercheDense,
    bm25_index: _RechercheSparse,
    reranker: _Reranker | None = None,
) -> HybridRetriever:
    """Assemble la recherche hybride à partir de ses composants déjà construits
    et du réglage du garde-fou étage 1.

    Le seuil de retrieval n'a de sens qu'avec un reclasseur : lui seul produit
    un score interprétable dans l'absolu (cf. retriever.py, en-tête). Sans
    reclasseur, il est ignoré plutôt que transmis à tort.
    """
    return HybridRetriever(
        embeddings=embeddings,
        recherche_dense=vector_store,
        recherche_sparse=bm25_index,
        reranker=reranker,
        seuil_rerank=settings.seuil_garde_fou_retrieval if reranker is not None else None,
    )


@trace("repondre")
def repondre(
    question: str,
    retriever: HybridRetriever,
    llm: _LLM,
    nli: _NLI,
    settings: Settings,
    k: int = 10,
) -> ReponseGeneree:
    """Orchestration complète d'une requête : recherche puis génération sous
    deux garde-fous (cf. CLAUDE.md, section garde-fou anti-hallucination).

    Le garde-fou étage 1 (seuil sur le score de reclassement) agit déjà dans
    retriever.search, avant même l'appel de ce module : ses résultats sont
    vides si aucun morceau n'est assez fiable, ce que generer_reponse traite
    comme un refus. Les étages suivants agissent ici, dans l'ordre :
      1. génération de la réponse (le LLM peut lui-même refuser) ;
      2. vérification des citations contre le texte réellement récupéré ;
      3. vérification de la fidélité de la réponse aux articles cités (NLI).
    """
    resultats = retriever.search(question, k=k)
    reponse = generer_reponse(question, resultats, llm)
    reponse = verifier_citations(reponse, resultats)
    return verifier_fidelite(
        reponse,
        resultats,
        nli,
        settings.seuil_nli_haut,
        settings.seuil_nli_bas,
        settings.seuil_nli_contradiction,
        llm=llm,
    )
