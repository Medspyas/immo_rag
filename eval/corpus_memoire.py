"""Construction du corpus en mémoire pour l'évaluation, sans base de données.

L'évaluation indexe les lois « cœur » puis interroge la recherche un grand
nombre de fois. Monter PostgreSQL pour cela n'apporte rien : les vecteurs des
articles tiennent largement en mémoire (quelques Mo). Ce module fait le même
découpage que le pipeline d'ingestion, mais range le résultat dans un tableau
numpy plutôt que dans pgvector.

Les embeddings sont calculés une seule fois puis mis en cache sur disque
(.cache/). Les exécutions suivantes relisent le cache : aucun calcul, aucun
usage CPU notable.

Contenu du fichier :
- FICHIERS_COEUR : les trois lois immo de référence.
- ResultatMemoire : une source retrouvée (forme attendue par HybridRetriever).
- RechercheDenseMemoire : recherche par le sens par produit scalaire numpy.
- CorpusMemoire : chunks + recherche dense + index BM25, prêts à interroger.
- construire_corpus : lit les fichiers, découpe, monte les index.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np

from app.rag.bm25_index import BM25Index
from app.rag.chunking import RawChunk, decouper_en_articles
from app.rag.contextualizer import ContextGenerator, contextualiser_chunks
from app.rag.embeddings import EmbeddingProvider
from app.rag.pipeline import documents_bm25, identifiant_chunk
from app.rag.vector_store import Chunk

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_CORPUS = RACINE / "data" / "clean"
CACHE = RACINE / ".cache"

# Corpus de référence : les lois directement utiles au métier, sans le code des
# impôts (plusieurs milliers d'articles, hors périmètre du premier checkpoint).
# La loi ALUR n'y figure pas : cf. TEXTES_NON_CONSOLIDES (scripts/ingerer_corpus.py).
FICHIERS_COEUR = [
    "loi_06_07_1989.txt",
    "Loi_hoguet.txt",
    "loi-n°65-557_du_10_juillet_1965.txt",
]


@dataclass
class ResultatMemoire:
    """Une source retrouvée. Les attributs id/doc_id/content/metadata sont ceux
    qu'attend HybridRetriever ; score porte la proximité de sens (cosinus)."""

    id: UUID
    doc_id: str
    content: str
    metadata: dict
    score: float


class RechercheDenseMemoire:
    """Recherche par le sens sans base : le vecteur de la question est multiplié
    à la matrice des vecteurs d'articles (tous normalisés), les meilleurs scores
    sont renvoyés. Le produit scalaire de deux vecteurs normalisés vaut leur
    similarité cosinus."""

    def __init__(self, chunks: list[Chunk], matrice: np.ndarray):
        self._chunks = chunks
        self._matrice = matrice

    def search(self, query_embedding: list[float], k: int) -> list[ResultatMemoire]:
        q = np.asarray(query_embedding, dtype=np.float32)
        scores = self._matrice @ q
        meilleurs = np.argsort(scores)[::-1][:k]
        return [
            ResultatMemoire(
                id=self._chunks[i].id,
                doc_id=self._chunks[i].doc_id,
                content=self._chunks[i].content,
                metadata=self._chunks[i].metadata,
                score=float(scores[i]),
            )
            for i in meilleurs
        ]


@dataclass
class CorpusMemoire:
    """Le corpus indexé en mémoire, prêt à être interrogé."""

    chunks: list[Chunk]
    embeddings: EmbeddingProvider
    dense: RechercheDenseMemoire
    bm25: BM25Index


def _articles_du_corpus(
    fichiers: list[str], dossier: Path, contextualizer: ContextGenerator | None = None
) -> list[RawChunk]:
    """Découpe chaque fichier de loi en articles (préambule et historique
    écartés par le découpage). Avec un contextualizer, situe les articles de
    chaque fichier dans leur document (cf. contextualizer.py) avant de les
    renvoyer — un appel au modèle par fichier."""
    articles: list[RawChunk] = []
    for nom in fichiers:
        chemin = dossier / nom
        texte = chemin.read_text(encoding="utf-8")
        arts = decouper_en_articles(texte, chemin.stem)
        if contextualizer is not None:
            arts = contextualiser_chunks(arts, texte, contextualizer)
        articles.extend(arts)
    return articles


def _matrice_embeddings(
    articles: list[RawChunk], embeddings: EmbeddingProvider, cache_dir: Path
) -> np.ndarray:
    """Renvoie la matrice des vecteurs d'articles, calculée en une passe la
    première fois puis relue depuis le cache disque.

    La clé de cache dépend du texte des articles : changer le corpus (ou le
    découpage) invalide automatiquement l'ancien cache.
    """
    cache_dir.mkdir(exist_ok=True)
    empreinte = hashlib.sha1(
        "".join(a.content for a in articles).encode("utf-8")
    ).hexdigest()[:16]
    fichier = cache_dir / f"embeddings_eval_{empreinte}.npy"
    if fichier.exists():
        return np.load(fichier)

    print(f"Calcul des embeddings pour {len(articles)} articles (une seule fois)…")
    vecteurs = np.asarray(
        embeddings.embed_documents([a.content for a in articles]), dtype=np.float32
    )
    np.save(fichier, vecteurs)
    return vecteurs


def construire_corpus(
    embeddings: EmbeddingProvider,
    *,
    fichiers: list[str] | None = None,
    dossier: Path | None = None,
    cache_dir: Path | None = None,
    contextualizer: ContextGenerator | None = None,
) -> CorpusMemoire:
    """Lit les fichiers de loi, les découpe en articles, calcule ou relit leurs
    embeddings, monte la recherche dense et l'index BM25.

    Sans contextualizer (par défaut) : articles indexés tels quels. Avec :
    chaque article est précédé du résumé de son document (un appel au modèle
    par fichier, cf. contextualizer.py) avant l'embedding — le cache disque
    distingue les deux (la clé dépend du texte réellement indexé)."""
    fichiers = fichiers or FICHIERS_COEUR
    dossier = dossier or DOSSIER_CORPUS
    cache_dir = cache_dir or CACHE

    articles = _articles_du_corpus(fichiers, dossier, contextualizer)
    matrice = _matrice_embeddings(articles, embeddings, cache_dir)

    chunks = [
        Chunk(
            id=identifiant_chunk(a.doc_id, a.article),
            doc_id=a.doc_id,
            content=a.content,
            embedding=vecteur.tolist(),
            metadata={
                "article": a.article, "abroge": a.abroge, "section": a.section, "titre": a.titre
            },
        )
        for a, vecteur in zip(articles, matrice, strict=True)
    ]

    return CorpusMemoire(
        chunks=chunks,
        embeddings=embeddings,
        dense=RechercheDenseMemoire(chunks, matrice),
        bm25=BM25Index(documents_bm25(chunks)),
    )
