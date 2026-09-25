"""Phase 1 de l'évaluation faithfulness : produit les traces à scorer.

Objectif (cf. CLAUDE.md, section "Garde-fou anti-hallucination") : comparer
notre garde-fou étage 2 en ligne (`verifier_fidelite`, NLI léger + repli LLM,
rapide) à la métrique `faithfulness` d'un outil d'évaluation indépendant
(décomposition en affirmations + vérification par un juge LLM, plus lente
mais faisant référence). Si les deux s'accordent, le garde-fou léger est
validé pour la production ; s'ils divergent souvent, il faut le revoir.

Ce script fait tourner la moitié "produit" (recherche + génération + les deux
étages du garde-fou), avec les MÊMES dépendances que l'app (embeddings locaux,
reclasseur, NLI, Groq) : il s'exécute dans le venv habituel du projet
(`.venv`), sans base de données (corpus en mémoire, comme ablation.py).

La moitié "mesure indépendante" (deepeval + Groq comme juge) est un module à
part, `score_faithfulness.py`, qui s'exécute dans un venv séparé
(`.venv-eval`) : deepeval entraîne des dépendances (client OpenAI, etc.) sans
rapport avec le produit, et pas de conflit de version à faire porter à
`generator.py`. Ce script-ci écrit donc un fichier intermédiaire (JSONL) que
l'autre relit — aucun des deux ne dépend du package de l'autre.

Lancement :
    python -m eval.run_eval                # tout le golden dataset (dans corpus)
    python -m eval.run_eval --limit 10      # se limiter aux 10 premiers cas
    python -m eval.run_eval --reprendre     # compléter une trace interrompue

Chaque trace est écrite dès qu'elle est prête : une interruption (quota
Groq, coupure) ne perd que le cas en cours. `--reprendre` saute alors les cas
déjà présents dans le fichier au lieu de les refaire, ce qui épargne le quota
Groq déjà consommé.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import Settings
from app.rag.embeddings import get_embedding_provider
from app.rag.generator import (
    _contextes_cites,
    generer_reponse,
    get_llm,
    verifier_citations,
    verifier_fidelite,
)
from app.rag.nli import get_nli_provider
from app.rag.reranker import get_rerank_provider
from app.rag.retriever import HybridRetriever
from eval.corpus_memoire import construire_corpus
from eval.golden import charger_golden

RACINE = Path(__file__).resolve().parent.parent
FICHIER_TRACES = RACINE / ".cache" / "eval_traces.jsonl"


def _trace(cas, resultats, reponse, verdict_local: str) -> dict:
    """Un cas du golden dataset, prêt à être écrit en JSONL : les champs dont
    score_faithfulness.py a besoin pour rejouer la mesure indépendante."""
    return {
        "id": cas.id,
        "question": cas.question,
        "reponse": reponse.reponse if reponse else "",
        "motif": reponse.motif if reponse else None,
        "contexts": _contextes_cites(reponse, resultats) if reponse and not reponse.refus else [],
        "verdict_local": verdict_local,
        "articles_attendus": [f"{cas.doc_id}:{a}" for a in cas.articles],
    }


def evaluer(cas_golden, retriever, llm, nli, settings: Settings) -> Iterator[dict]:
    """Fait tourner recherche + génération + garde-fou étage 2 sur chaque cas,
    et produit une trace par cas (voir `_trace`) au fur et à mesure, pour
    qu'elle puisse être écrite avant de passer au cas suivant."""
    for cas in cas_golden:
        resultats = retriever.search(cas.question, k=10)
        if not resultats:
            yield _trace(cas, resultats, None, "refus_retrieval")
            continue

        reponse_brute = generer_reponse(cas.question, resultats, llm)
        if reponse_brute.refus:
            yield _trace(cas, resultats, reponse_brute, "refus_generation")
            continue

        reponse_citations = verifier_citations(reponse_brute, resultats)
        if reponse_citations.refus:
            yield _trace(cas, resultats, reponse_citations, "refus_citation")
            continue

        # reponse_citations est ce qu'on confronte à la mesure indépendante :
        # le texte tel que rédigé, avant que notre propre garde-fou étage 2 ne
        # le transforme éventuellement en refus.
        reponse_apres = verifier_fidelite(
            reponse_citations,
            resultats,
            nli,
            settings.seuil_nli_haut,
            settings.seuil_nli_bas,
            settings.seuil_nli_contradiction,
            llm,
        )
        verdict = "refus_fidelite" if reponse_apres.refus else "fondee"
        yield _trace(cas, resultats, reponse_citations, verdict)


def traces_existantes(fichier: Path) -> list[dict]:
    """Traces valides d'un fichier existant, pour `--reprendre`. Une ligne
    illisible (écriture coupée net au milieu d'une ligne) est écartée : le
    cas correspondant sera simplement refait."""
    if not fichier.exists():
        return []
    traces = []
    for ligne in fichier.read_text(encoding="utf-8").splitlines():
        try:
            trace = json.loads(ligne)
        except json.JSONDecodeError:
            continue
        if isinstance(trace, dict) and "id" in trace:
            traces.append(trace)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="limiter aux N premiers cas")
    parser.add_argument(
        "--reprendre", action="store_true", help="compléter la trace existante au lieu de l'écraser"
    )
    args = parser.parse_args()

    settings = Settings()
    embeddings = get_embedding_provider(settings)
    reranker = get_rerank_provider(settings)
    llm = get_llm(settings)
    nli = get_nli_provider(settings)

    print("Préparation du corpus en mémoire…")
    corpus = construire_corpus(embeddings)
    retriever = HybridRetriever(
        embeddings, corpus.dense, corpus.bm25, reranker=reranker,
        seuil_rerank=settings.seuil_garde_fou_retrieval,
    )

    cas_golden = [c for c in charger_golden() if not c.hors_corpus]
    if args.limit:
        cas_golden = cas_golden[: args.limit]

    traces = traces_existantes(FICHIER_TRACES) if args.reprendre else []
    deja_faits = {t["id"] for t in traces}
    a_faire = [c for c in cas_golden if c.id not in deja_faits]
    if deja_faits:
        print(f"Reprise : {len(deja_faits)} cas déjà tracés, conservés.")

    print(f"Génération de {len(a_faire)} traces (appels Groq + NLI réels)…")
    FICHIER_TRACES.parent.mkdir(exist_ok=True)
    # Le fichier est réécrit avec les seules traces valides conservées, puis
    # chaque nouvelle trace y est ajoutée et vidée sur disque aussitôt.
    with FICHIER_TRACES.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
        for i, t in enumerate(evaluer(a_faire, retriever, llm, nli, settings), start=1):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            f.flush()
            traces.append(t)
            print(f"  [{i}/{len(a_faire)}] {t['id']} : {t['verdict_local']}")

    comptes: dict[str, int] = {}
    for t in traces:
        comptes[t["verdict_local"]] = comptes.get(t["verdict_local"], 0) + 1
    print(f"\nTraces écrites dans {FICHIER_TRACES}")
    for verdict, n in sorted(comptes.items()):
        print(f"  {verdict:<18} {n}")
    print(
        "\nSuite : score_faithfulness.py (venv .venv-eval) pour la mesure "
        "indépendante et la comparaison."
    )


if __name__ == "__main__":
    main()
