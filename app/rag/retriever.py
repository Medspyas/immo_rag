"""
Retriever hybride : combine la recherche dense (vector_store.py, par sens) et
la recherche par mots-clés (bm25_index.py) via fusion RRF, puis reranking.

Étape 1 (ce module pour l'instant) : uniquement la fusion RRF, en fonction pure
indépendante des deux modules de recherche — le branchement (appeler les deux
recherches puis fusionner leurs résultats) viendra dans une prochaine étape.
"""
from __future__ import annotations

from uuid import UUID


def fusion_rrf(classements: list[list[UUID]], k: int = 60) -> list[UUID]:
    """Fusionne plusieurs classements d'ids en un seul, par Reciprocal Rank Fusion.

    Analogie : la recherche dense et BM25 donnent parfois des avis très
    différents sur un même chunk ("2e pour l'une, 8e pour l'autre") — impossible
    de comparer directement leurs scores, qui n'ont pas la même échelle
    (similarité cosinus entre 0 et 1 vs score BM25 brut non borné). RRF évite le
    problème en ne regardant que le *rang* de chaque document dans chaque
    classement, jamais son score.

    Formule : score(d) = somme, sur chaque classement qui contient d, de
    1 / (k + rang(d)). `k` (60, valeur d'origine du papier RRF) amortit l'écart
    entre les tout premiers rangs : un 1er bat nettement un 2e, mais un 50e ne
    bat que de peu un 51e.

    Un document absent d'un classement ne reçoit simplement aucune contribution
    de ce classement — pas de pénalité, juste rien à ajouter.
    """
    scores: dict[UUID, float] = {}
    for classement in classements:
        for rang, doc_id in enumerate(classement, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rang)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
