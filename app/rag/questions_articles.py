"""Questions auxquelles chaque article répond (doc2query), pour la recherche
par mots-clés.

Une question et l'article qui y répond n'emploient pas toujours les mêmes mots :
« un mandat peut-il être à durée indéterminée ? » face à « comporter une
limitation de ses effets dans le temps » (article 7 de la loi Hoguet). Un modèle
de langage a donc rédigé, pour chaque article des trois lois immobilières,
quatre questions dans le vocabulaire courant de la personne qui les pose. Elles
sont ajoutées à l'index BM25 de l'article et montrées au reclasseur devant
son texte (cf. retriever.PREFIXE_QUESTIONS), jamais au texte lui-même : le LLM
et la vérification des citations ne voient que le texte de loi.

Questions générées une fois, hors ligne (qwen3.8-27b via Groq, offre gratuite),
et livrées avec le code dans questions_articles.jsonl. Aucun appel de modèle à
l'ingestion ni au démarrage.

Mesuré le 2026-10-14 sur le golden dataset (corpus complet, 40 questions) :
MRR 0,686 → 0,723, hit@1 55 % → 60 %, hit@3 87,5 % → 90 %, scores hors corpus
inchangés. Écartés : vectoriser les questions pour la recherche par le sens
(hit@3 85 %, des questions perdues ailleurs). Les montrer au reclasseur, d'abord
écarté (hit@3 82,5 %), a été repris le 2026-10-15 avec le titre de la loi
devant : MRR 0,725 → 0,765, hit@5 90 % → 92,5 % (cf. retriever.py).

Contenu du fichier :
- charger_questions : lit le fichier, indexé par (loi, article).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FICHIER_QUESTIONS = Path(__file__).with_name("questions_articles.jsonl")


@lru_cache(maxsize=1)
def charger_questions(chemin: Path = FICHIER_QUESTIONS) -> dict[tuple[str, str], list[str]]:
    """Questions de chaque article, par (doc_id, numéro d'article). Fichier
    absent : dictionnaire vide, la recherche fonctionne alors sans."""
    if not chemin.exists():
        return {}
    questions: dict[tuple[str, str], list[str]] = {}
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            d = json.loads(ligne)
            questions[(d["doc_id"], d["article"])] = d["questions"]
    return questions
