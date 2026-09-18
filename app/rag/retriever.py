"""Recherche hybride : combinaison de la recherche par le sens et par mots-clés.

Les deux recherches sont exécutées : celle par le sens (vector_store.py) et
celle par mots-clés (bm25_index.py). Leurs deux classements sont ensuite
fusionnés en un seul. Les scores des deux méthodes n'étant pas comparables
(échelles différentes), la fusion ne tient compte que du rang de chaque
document dans chaque classement. C'est le principe de la méthode RRF
(Reciprocal Rank Fusion).

Construction du module en trois temps (cf. CLAUDE.md) :
  1. la fusion seule (fusion_rrf)          [fait]
  2. le raccordement des deux recherches   [fait]
  3. le reclassement fin par-dessus        [ici, via reranker.py]

Le reclassement est facultatif. Sans reclasseur, l'ordre des résultats est
celui de la fusion. Avec un reclasseur, les meilleurs candidats de la fusion
sont soumis à un cross-encoder (reranker.py) qui relit la question et chaque
texte ensemble ; l'ordre final suit alors le score du cross-encoder.

Garde-fou étage 1 (cf. CLAUDE.md) : un morceau dont le score de reclassement
tombe sous seuil_rerank est écarté des résultats, avant même d'atteindre le
LLM générateur. C'est le seul score des trois (RRF, dense, reclassement) dont
l'échelle est interprétable dans l'absolu (0 à 1, cross-encoder) ; le score
RRF et le cosinus dense ne le sont pas (mesuré : le cosinus dense des
questions hors corpus chevauche celui des questions du corpus). Sans
reclasseur, ce filtre n'a pas d'effet : il n'existe alors aucun score sur
lequel le fonder.

Contenu du fichier :
- fusion_rrf : fusionne plusieurs classements d'identifiants en un seul.
- RetrievalResult : un morceau retrouvé après fusion, avec son rang dans
  chaque recherche et, le cas échéant, son score de reclassement.
- HybridRetriever : enchaîne les deux recherches, leur fusion, le
  reclassement facultatif et le garde-fou de retrieval.
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


# --- Étapes 2 et 3 : les deux recherches, leur fusion, le reclassement ------


class _ResultatRecherche(Protocol):
    """Partie commune aux résultats des deux recherches : de quoi identifier un
    morceau et le restituer. Les scores des deux recherches ne sont pas utilisés
    ici ; la fusion ne considère que les rangs."""

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


class _Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]: ...


class RetrievalResult(BaseModel):
    """Un morceau retrouvé par la recherche hybride.

    score_rrf ordonne les morceaux issus de la fusion. rang_dense et rang_bm25
    indiquent le rang du morceau dans chaque recherche, ou None s'il n'y
    figurait pas : cette information retrace l'origine du résultat.

    score_rerank est le score du cross-encoder (entre 0 et 1), présent seulement
    lorsqu'un reclasseur a été fourni. C'est lui qui portera le garde-fou de
    retrieval ; il vaut None en son absence.
    """

    id: UUID
    doc_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    score_rrf: float
    rang_dense: int | None = None
    rang_bm25: int | None = None
    score_rerank: float | None = None


class HybridRetriever:
    """Enchaîne les deux recherches, leur fusion et un reclassement facultatif.

    Les composants utilisés (embeddings, recherche par le sens, recherche par
    mots-clés, reclasseur) sont fournis en paramètres, non instanciés ici : le
    module reste testable sans base ni modèle. pipeline.py les raccordera depuis
    .env.

    k_par_source : nombre de morceaux demandés à chaque recherche avant fusion.
    Ce nombre dépasse le total final afin qu'un morceau mal classé par une
    recherche mais bien classé par l'autre puisse remonter.

    reranker : reclasseur facultatif. Fourni, il réordonne les meilleurs
    candidats de la fusion. Absent (None), l'ordre reste celui de la fusion.

    k_avant_rerank : nombre de candidats de la fusion soumis au reclasseur. Le
    cross-encoder traite chaque candidat séparément ; ce plafond limite son
    coût. Sans effet lorsque reranker vaut None.

    seuil_rerank : garde-fou étage 1. Fourni avec un reclasseur, tout candidat
    dont le score de reclassement tombe sous ce seuil est écarté des résultats
    plutôt que renvoyé. Sans effet lorsque reranker vaut None (aucun score de
    reclassement à comparer).
    """

    def __init__(
        self,
        embeddings: _Embeddings,
        recherche_dense: _RechercheDense,
        recherche_sparse: _RechercheSparse,
        reranker: _Reranker | None = None,
        k_par_source: int = 20,
        k_rrf: int = 60,
        k_avant_rerank: int = 20,
        seuil_rerank: float | None = None,
    ):
        self._embeddings = embeddings
        self._dense = recherche_dense
        self._sparse = recherche_sparse
        self._reranker = reranker
        self._k_par_source = k_par_source
        self._k_rrf = k_rrf
        self._k_avant_rerank = k_avant_rerank
        self._seuil_rerank = seuil_rerank

    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Les k morceaux les mieux placés pour la question.

        L'ordre vient de la fusion RRF, ou du reclasseur lorsqu'il est fourni.
        Renvoie une liste vide lorsque les deux recherches ne renvoient rien.
        """
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

        scores = _scores_rrf([list(rang_dense), list(rang_bm25)], k=self._k_rrf)
        ids_ordonnes = sorted(scores, key=lambda i: scores[i], reverse=True)

        def _resultat(chunk_id: UUID, score_rerank: float | None = None) -> RetrievalResult:
            source = par_id[chunk_id]
            return RetrievalResult(
                id=chunk_id,
                doc_id=source.doc_id,
                content=source.content,
                metadata=source.metadata,
                score_rrf=scores[chunk_id],
                rang_dense=rang_dense.get(chunk_id),
                rang_bm25=rang_bm25.get(chunk_id),
                score_rerank=score_rerank,
            )

        if self._reranker is None:
            return [_resultat(chunk_id) for chunk_id in ids_ordonnes[:k]]

        # Seuls les meilleurs candidats de la fusion sont soumis au reclasseur ;
        # leur ordre final est celui qu'il renvoie, pas celui de la fusion.
        candidats = ids_ordonnes[: self._k_avant_rerank]
        textes = [par_id[chunk_id].content for chunk_id in candidats]
        classement = self._reranker.rerank(query, textes)

        # Garde-fou étage 1 : un candidat sous le seuil n'est pas assez fiable
        # pour être soumis au LLM générateur, refus (résultats vides) plutôt
        # qu'un appel de génération inutile.
        if self._seuil_rerank is not None:
            classement = [
                (indice, score) for indice, score in classement if score >= self._seuil_rerank
            ]

        return [
            _resultat(candidats[indice], score_rerank=score)
            for indice, score in classement[:k]
        ]
