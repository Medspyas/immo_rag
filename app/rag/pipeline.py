"""Chaîne d'ingestion : d'un texte de loi aux morceaux prêts à rechercher.

Les briques précédentes traitent chacune une étape isolée (découpage, contexte,
embeddings, rangement). Ce module les met bout à bout : un texte de loi entre,
des morceaux indexés sortent.

Étapes enchaînées (cf. CLAUDE.md, section Flux) :
  1. découper le texte en articles            (chunking.py)
  2. situer chaque article dans la loi         (contextualizer.py, facultatif)
  3. transformer chaque article en vecteur,
     plus un par fenêtre d'un article long     (embeddings.py, fenetres.py)
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
- TEXTES_NON_CONSOLIDES, fichiers_du_corpus : textes de data/clean/ indexés,
  liste commune à l'ingestion réelle et à l'évaluation.
- identifiant_chunk : identifiant stable d'un article — même loi et même numéro
  donnent toujours le même identifiant, d'une ingestion à l'autre.
- preparer_morceaux : étapes 1 à 3, sans toucher à la base.
- ingerer_document : les quatre étapes, rangement en base compris.
- documents_bm25 : convertit des morceaux préparés en entrées pour l'index BM25,
  avec les questions de chaque article (questions_articles.py).
- construire_index_bm25 : reconstruit l'index BM25 à partir de la base.
- construire_retriever : assemble la recherche hybride à partir de ses
  composants et du réglage du garde-fou étage 1.
- repondre : orchestration complète d'une requête (recherche + génération +
  garde-fous), tracée dans Langfuse quand configuré (observability/tracing.py).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel

from app.config import Settings
from app.observability.tracing import trace
from app.rag.bm25_index import BM25Document, BM25Index
from app.rag.chunking import decouper_en_articles
from app.rag.contextualizer import ContextGenerator, contextualiser_chunks
from app.rag.fenetres import fenetres_article
from app.rag.generator import (
    ReponseGeneree,
    completude_a_verifier,
    corriger_citations,
    generer_reponse,
    verifier_citations,
    verifier_completude,
    verifier_fidelite,
)
from app.rag.nli import ScoreNLI
from app.rag.questions_articles import charger_questions
from app.rag.retriever import HybridRetriever
from app.rag.suivi import Echange, reformuler_question
from app.rag.vector_store import Chunk, MorceauStocke

# Espace de noms fixe pour dériver l'identifiant d'un article de façon
# déterministe. Conséquence : réingérer la même loi réécrit les mêmes morceaux
# (upsert) au lieu d'en créer des doublons.
NS_CHUNK = UUID("00000000-0000-0000-0000-000000000001")

# Lois modificatives en version Journal officiel, non indexées. Leurs articles
# ne sont pas du droit autonome mais des consignes de modification d'autres
# textes (« Après l'article 24-1 de la loi n° 89-462, il est inséré… »), figées
# à la date de publication. Leur effet se lit dans les versions consolidées des
# textes modifiés, déjà présentes dans le corpus. Les indexer fait citer du droit
# périmé : la garantie universelle des loyers d'ALUR passait devant l'article 24
# de la loi de 1989 pour une question sur les impayés. Mesuré sur le golden sans
# ALUR : MRR 0,686 → 0,706, hit@3 81 % → 87 %.
TEXTES_NON_CONSOLIDES = {"loi_ALUR.txt"}


def fichiers_du_corpus(dossier: Path) -> list[str]:
    """Noms des textes de dossier indexés en production : tous les .txt sauf
    TEXTES_NON_CONSOLIDES. L'évaluation lit la même liste, pour mesurer le
    système réellement interrogé et non un sous-ensemble plus facile."""
    return sorted(
        p.name for p in dossier.glob("*.txt") if p.name not in TEXTES_NON_CONSOLIDES
    )


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

    def tous(self) -> list[MorceauStocke]: ...


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
    # Fenêtres des articles longs (fenetres.py), vectorisées en un seul appel
    # puis rendues à leur article.
    fenetres = [fenetres_article(a.content) for a in articles]
    vecteurs_fenetres = iter(
        embeddings.embed_documents([f for liste in fenetres for f in liste])
        if any(fenetres)
        else []
    )

    return [
        Chunk(
            id=identifiant_chunk(doc_id, article.article),
            doc_id=doc_id,
            content=article.content,
            embedding=vecteur,
            metadata={
                "article": article.article,
                "abroge": article.abroge,
                "section": article.section,
                "titre": article.titre,
            },
            fenetres=[next(vecteurs_fenetres) for _ in liste],
        )
        for article, vecteur, liste in zip(articles, vecteurs, fenetres, strict=True)
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


def documents_bm25(
    morceaux: Sequence[Chunk | MorceauStocke],
    questions: dict[tuple[str, str], list[str]] | None = None,
) -> list[BM25Document]:
    """Convertit des morceaux (préparés ou relus en base) en entrées pour
    l'index BM25, qui n'a pas besoin des vecteurs. Les questions de chaque
    article (questions_articles.py) sont indexées avec lui, sans entrer dans
    son texte."""
    questions = charger_questions() if questions is None else questions
    return [
        BM25Document(
            id=m.id,
            doc_id=m.doc_id,
            content=m.content,
            metadata=m.metadata,
            complement="\n".join(questions.get((m.doc_id, m.metadata.get("article")), [])),
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
    """Index BM25 : recherche par mots-clés, et article retrouvé par son
    numéro pour suivre les renvois (cf. renvois.py)."""

    def search(self, query: str, k: int) -> list: ...

    def article(self, famille: str, numero: str): ...


class _Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]: ...


def construire_index_bm25(
    vector_store: _VectorStoreLecture,
    questions: dict[tuple[str, str], list[str]] | None = None,
) -> BM25Index:
    """Reconstruit l'index BM25 (en mémoire) à partir de ce qui est déjà rangé
    en base.

    Utile au démarrage de l'application : contrairement à l'ingestion, où les
    morceaux sortent de preparer_morceaux, ici on repart de la base — c'est
    elle qui fait foi sur ce qui a été ingéré.

    Même conversion que l'évaluation (documents_bm25), questions des articles
    comprises : jusqu'au 2026-10-16, l'index de production en était privé
    alors que le gate les mesurait (q28 et q31, loi Hoguet, perdues en base).
    """
    return BM25Index(documents_bm25(vector_store.tous(), questions))


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
        k_par_source=settings.k_par_source,
        k_avant_rerank=settings.k_avant_rerank,
        seuil_rerank=settings.seuil_garde_fou_retrieval if reranker is not None else None,
        index_articles=bm25_index,
    )


@trace("repondre")
def repondre(
    question: str,
    retriever: HybridRetriever,
    llm: _LLM,
    nli: _NLI,
    settings: Settings,
    k: int | None = None,
    juge: _LLM | None = None,
    historique: list[Echange] | None = None,
) -> ReponseGeneree:
    """Orchestration complète d'une requête : recherche puis génération sous
    deux garde-fous (cf. CLAUDE.md, section garde-fou anti-hallucination).

    Le garde-fou étage 1 (seuil sur le score de reclassement) agit déjà dans
    retriever.search, avant même l'appel de ce module : ses résultats sont
    vides si aucun morceau n'est assez fiable, ce que generer_reponse traite
    comme un refus. Les étages suivants agissent ici, dans l'ordre :
      1. génération de la réponse (le LLM peut lui-même refuser) ;
      2. vérification des citations contre le texte réellement récupéré,
         après une seconde chance unique si certaines sont introuvables ;
      3. vérification de la fidélité de la réponse aux articles cités (NLI).

    k fixe le nombre d'articles envoyés au LLM ; par défaut
    settings.k_articles_generation. juge sert de juge de repli pour la
    fidélité ; par défaut llm.

    historique : les échanges précédents de la conversation. Une question de
    suivi est d'abord réécrite en question autonome (suivi.py), qui sert
    ensuite à la recherche et à la génération.
    """
    if k is None:
        k = settings.k_articles_generation
    question_posee = question
    question = reformuler_question(question, historique or [], llm)
    reponse = _repondre_question(question, retriever, llm, nli, settings, k, juge)
    if question != question_posee:
        reponse = reponse.model_copy(update={"question_comprise": question})
    return reponse


def _repondre_question(
    question: str,
    retriever: HybridRetriever,
    llm: _LLM,
    nli: _NLI,
    settings: Settings,
    k: int,
    juge: _LLM | None,
) -> ReponseGeneree:
    """Recherche, génération et garde-fous pour une question autonome.

    Une réponse vérifiée que le contrôle de complétude juge trompeuse par
    omission est rédigée une seconde fois, l'élément omis signalé au LLM, et
    repasse les mêmes garde-fous. Une seule fois : chaque tour consomme du
    quota gratuit.

    Si la seconde rédaction échoue aux garde-fous, la première est gardée :
    déjà vérifiée phrase par phrase, elle vaut mieux qu'un refus. Le contrôle
    signale aussi des omissions qui n'en sont pas, et la seconde rédaction
    transformait alors une bonne réponse en refus (q16, 2026-10-21)."""
    resultats = retriever.search(question, k=k)
    reponse = _rediger_et_verifier(question, resultats, llm, nli, settings, juge)
    if not _completude_a_controler(reponse, resultats, settings):
        return reponse
    omission = verifier_completude(question, reponse, resultats, juge or llm)
    if omission is None:
        return reponse
    seconde = _rediger_et_verifier(question, resultats, llm, nli, settings, juge, omission)
    return reponse if seconde.refus else seconde


def _completude_a_controler(reponse: ReponseGeneree, resultats: list, settings: Settings) -> bool:
    """Le contrôle de complétude s'applique-t-il à cette réponse, selon le
    réglage controle_completude ?"""
    if settings.controle_completude == "non":
        return False
    if settings.controle_completude == "toujours":
        return not reponse.refus
    return completude_a_verifier(reponse, resultats)


def _rediger_et_verifier(
    question: str,
    resultats: list,
    llm: _LLM,
    nli: _NLI,
    settings: Settings,
    juge: _LLM | None,
    omission: str | None = None,
) -> ReponseGeneree:
    """Rédaction de la réponse puis garde-fous : citations, puis fidélité."""
    reponse = generer_reponse(question, resultats, llm, omission)
    reponse = corriger_citations(question, resultats, llm, reponse)
    reponse = verifier_citations(reponse, resultats)
    return verifier_fidelite(
        reponse,
        resultats,
        nli,
        settings.seuil_nli_haut,
        settings.seuil_nli_contradiction,
        llm=juge or llm,
    )
