"""Mesures de qualité de la recherche (retrieval).

Ces fonctions ne connaissent ni la base ni les modèles : elles reçoivent, pour
chaque question, la liste ordonnée des sources renvoyées et l'ensemble des
sources attendues, puis calculent des scores. Une source est identifiée par un
couple (doc_id, article).

Deux mesures :
- hit@k : la question est réussie si une source attendue figure dans les k
  premiers résultats. Répond à « le bon article est-il remonté ? ».
- MRR (Mean Reciprocal Rank) : moyenne de 1/rang de la première source
  attendue. Récompense le fait de placer le bon article haut, pas seulement
  dans les k premiers. 1,0 = toujours en tête ; 0,0 = jamais trouvé.

Contenu du fichier :
- Source : alias de (doc_id, article).
- rang_premier_pertinent : position de la première source attendue, ou None.
- hit_at_k : la première source attendue est-elle dans les k premiers.
- taux_hit_at_k : proportion de questions réussies à k, sur un lot.
- mrr : Mean Reciprocal Rank sur un lot.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

Source = tuple[str, str]


def rang_premier_pertinent(
    resultats: Sequence[Source], cibles: Iterable[Source]
) -> int | None:
    """Position (à partir de 1) de la première source attendue dans les
    résultats, ou None si aucune n'y figure."""
    cibles = set(cibles)
    for position, source in enumerate(resultats, start=1):
        if source in cibles:
            return position
    return None


def hit_at_k(rang: int | None, k: int) -> bool:
    """Vrai si une source attendue a été trouvée au rang k ou avant."""
    return rang is not None and rang <= k


def taux_hit_at_k(rangs: Iterable[int | None], k: int) -> float:
    """Proportion de questions pour lesquelles une source attendue est dans les
    k premiers résultats. Lot vide : 0,0."""
    rangs = list(rangs)
    if not rangs:
        return 0.0
    return sum(hit_at_k(r, k) for r in rangs) / len(rangs)


def mrr(rangs: Iterable[int | None]) -> float:
    """Moyenne des rangs réciproques (1/rang), 0 quand rien n'est trouvé. Lot
    vide : 0,0."""
    rangs = list(rangs)
    if not rangs:
        return 0.0
    return sum((1.0 / r if r else 0.0) for r in rangs) / len(rangs)
