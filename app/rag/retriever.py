"""Recherche hybride : combinaison de la recherche par le sens et par mots-clés.

Les deux recherches sont exécutées : celle par le sens (vector_store.py) et
celle par mots-clés (bm25_index.py). Leurs deux classements sont ensuite
fusionnés en un seul. Les scores des deux méthodes n'étant pas comparables
(échelles différentes), la fusion ne tient compte que du rang de chaque
document dans chaque classement. C'est le principe de la méthode RRF
(Reciprocal Rank Fusion).

Construction du module en trois temps (cf. CLAUDE.md) :
  1. la fusion seule (fusion_rrf)          [fait]
  2. le raccordement des deux recherches   [ici]
  3. le reclassement fin par-dessus        [à venir]

Le garde-fou "refus en l'absence de résultat suffisamment pertinent" est prévu
à l'étape 3, sur le score du reclasseur : le score de fusion n'offre pas de
seuil interprétable.

Contenu du fichier :
- fusion_rrf : fusionne plusieurs classements d'identifiants en un seul.
- RetrievalResult : un morceau retrouvé après fusion, avec son rang dans
  chaque recherche (utile pour retracer son origine).
- HybridRetriever : enchaîne les deux recherches et leur fusion.
"""
from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, Field


def _scores_rrf(classements: list[list[UUID]], k: int = 60) -> dict[UUID, float]:
    """Calcule le score de fusion de chaque identifiant.

    Pour chaque classement où il figure, l'identifiant reçoit 1 / (k + rang) :
    un meilleur rang rapporte davantage. k (60, valeur d'origine de la méthode)
    atténue l'écart entre les premières places. Un identifiant absent d'un
    classement n'en reçoit rien, sans pénalité.
    """
    scores: dict[UUID, float] = {}
    for classement in classements:
        for rang, doc_id in enumerate(classement, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rang)
    return scores


def fusion_rrf(classements: list[list[UUID]], k: int = 60) -> list[UUID]:
    """Fusionne plusieurs classements d'identifiants en un seul.

    Les deux recherches classent parfois différemment un même morceau, et leurs
    scores ne sont pas comparables. La fusion ne considère donc que le rang de
    chaque morceau dans chaque classement, jamais les scores.
    """
    scores = _scores_rrf(classements, k)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


# --- Étape 2 : les deux recherches et leur fusion ---------------------------


class _ResultatRecherche(Protocol):
    """Partie commune aux résultats des deux recherches : de quoi identifier un
    morceau et le restituer. Les scores ne sont pas utilisés ici ; la fusion ne
    considère que les rangs."""

    id: UUID
    doc_id: str
    content: str
    metadata: dict


class _RechercheDense(Protocol):
    def search(self, query_embedding: list[float], k: int) -> list[_ResultatRecherche]: ...


class _RechercheSparse(Protocol):
    def search(self, query: str, k: int) -> list[_ResultatRecherche]: ...


class _Embeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class RetrievalResult(BaseModel):
    """Un morceau retrouvé par la recherche hybride, après fusion.

    score_rrf ordonne les morceaux entre eux, sans seuil absolu (aucun
    garde-fou à ce stade). rang_dense et rang_bm25 indiquent le rang du morceau
    dans chaque recherche, ou None s'il n'y figurait pas : cette information
    retrace l'origine du résultat.
    """

    id: UUID
    doc_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    score_rrf: float
    rang_dense: int | None = None
    rang_bm25: int | None = None


class HybridRetriever:
    """Enchaîne les deux recherches puis leur fusion.

    Les composants utilisés (embeddings, recherche par le sens, recherche par
    mots-clés) sont fournis en paramètres, non instanciés ici : le module reste
    testable sans base ni modèle. pipeline.py les raccordera depuis .env.

    k_par_source : nombre de morceaux demandés à chaque recherche avant fusion.
    Ce nombre dépasse le total final afin qu'un morceau mal classé par une
    recherche mais bien classé par l'autre puisse remonter.
    """

    def __init__(
        self,
        embeddings: _Embeddings,
        recherche_dense: _RechercheDense,
        recherche_sparse: _RechercheSparse,
        k_par_source: int = 20,
        k_rrf: int = 60,
    ):
        self._embeddings = embeddings
        self._dense = recherche_dense
        self._sparse = recherche_sparse
        self._k_par_source = k_par_source
        self._k_rrf = k_rrf

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Les k morceaux les mieux placés pour la question après fusion.
        Renvoie une liste vide lorsque les deux recherches ne renvoient rien."""
        embedding_requete = self._embeddings.embed_query(query)
        resultats_denses = self._dense.search(embedding_requete, k=self._k_par_source)
        resultats_sparses = self._sparse.search(query, k=self._k_par_source)

        # Un morceau peut provenir des deux recherches ; le contenu étant
        # identique, seule la première occurrence est conservée.
        par_id: dict[UUID, _ResultatRecherche] = {}
        for resultat in [*resultats_denses, *resultats_sparses]:
            par_id.setdefault(resultat.id, resultat)

        rang_dense = {r.id: i for i, r in enumerate(resultats_denses, start=1)}
        rang_bm25 = {r.id: i for i, r in enumerate(resultats_sparses, start=1)}

        scores = _scores_rrf(
            [list(rang_dense), list(rang_bm25)],
            k=self._k_rrf,
        )
        ids_ordonnes = sorted(scores, key=lambda i: scores[i], reverse=True)

        return [
            RetrievalResult(
                id=chunk_id,
                doc_id=par_id[chunk_id].doc_id,
                content=par_id[chunk_id].content,
                metadata=par_id[chunk_id].metadata,
                score_rrf=scores[chunk_id],
                rang_dense=rang_dense.get(chunk_id),
                rang_bm25=rang_bm25.get(chunk_id),
            )
            for chunk_id in ids_ordonnes[:k]
        ]
