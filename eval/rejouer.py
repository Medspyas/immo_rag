"""Rejoue les garde-fous hors-ligne sur des traces déjà produites par run_eval.py.

Le LLM ne répond pas deux fois exactement pareil (même à température 0) : une
nouvelle génération Groq ne permet donc pas de savoir si un changement du
garde-fou améliore les choses ou si le LLM a simplement répondu autrement.
Ce script relit la réponse brute enregistrée dans chaque trace (`generation`)
et les articles récupérés (`articles`), puis refait passer exactement cette
réponse dans verifier_citations et verifier_fidelite : même entrée, résultat
comparable d'un réglage à l'autre, sans consommer de quota Groq.

Le juge LLM de repli n'est pas appelé : un juge factice note seulement qu'il
aurait été consulté (verdict « a_juger »), ce qui garde le rejeu déterministe.

Lancement :
    python -m eval.rejouer .cache/traces_k5_rejouables.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import Settings
from app.rag.generator import (
    ReponseGeneree,
    reparer_citations,
    sans_exemples,
    verifier_citations,
    verifier_fidelite,
)
from app.rag.nli import get_nli_provider


class _JugeFactice:
    """Tient lieu de juge LLM : répond « fondée » et compte les consultations,
    pour distinguer une réponse validée par le seul NLI d'une réponse qui
    aurait dépendu du juge."""

    def __init__(self) -> None:
        self.appels = 0

    def generer(self, consigne: str, question: str) -> str:
        self.appels += 1
        liste = question.split("Affirmations à vérifier :")[-1]
        n = len(re.findall(r"^\d+\. ", liste, re.MULTILINE))
        return json.dumps({"fondees": [True] * n})


def rejouer_trace(trace: dict, nli, settings: Settings) -> tuple[str, str | None]:
    """Verdict des garde-fous sur la réponse enregistrée d'une trace, et motif
    du refus éventuel. Une trace sans génération (refus de la recherche) garde
    son verdict d'origine."""
    if not trace.get("generation"):
        return trace["verdict_local"], trace.get("motif")

    reponse = sans_exemples(ReponseGeneree.model_validate(trace["generation"]))
    if reponse.refus:
        return "refus_generation", reponse.motif

    articles = [SimpleNamespace(**a) for a in trace["articles"]]
    apres_citations = verifier_citations(reparer_citations(reponse, articles), articles)
    if apres_citations.refus:
        return "refus_citation", apres_citations.motif

    juge = _JugeFactice()
    apres_fidelite = verifier_fidelite(
        apres_citations,
        articles,
        nli,
        settings.seuil_nli_haut,
        settings.seuil_nli_contradiction,
        juge,
    )
    if apres_fidelite.refus:
        return "refus_fidelite", apres_fidelite.motif
    return ("a_juger" if juge.appels else "fondee"), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="fichier de traces JSONL (run_eval.py)")
    args = parser.parse_args()

    settings = Settings()
    nli = get_nli_provider(settings)

    comptes: dict[str, int] = {}
    for ligne in args.traces.read_text(encoding="utf-8").splitlines():
        trace = json.loads(ligne)
        verdict, motif = rejouer_trace(trace, nli, settings)
        comptes[verdict] = comptes.get(verdict, 0) + 1
        change = "" if verdict == trace["verdict_local"] else f"  (en ligne : {trace['verdict_local']})"
        print(f"{trace['id']:<5} {verdict:<17}{change}")
        if motif and verdict.startswith("refus_"):
            print(f"      {motif[:160]}")

    print()
    for verdict, n in sorted(comptes.items()):
        print(f"  {verdict:<17} {n}")


if __name__ == "__main__":
    main()
