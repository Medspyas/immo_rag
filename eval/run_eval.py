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
    python -m eval.run_eval --k 5 --sortie .cache/traces_k5.jsonl
                                            # 5 articles envoyés au LLM au lieu de 10

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
    corriger_citations,
    generer_reponse,
    get_llm,
    get_llm_juge,
    verifier_citations,
    verifier_completude,
    verifier_fidelite,
)
from app.rag.nli import get_nli_provider
from app.rag.pipeline import _completude_a_controler
from app.rag.reranker import get_rerank_provider
from app.rag.retriever import HybridRetriever
from app.rag.suivi import reformuler_question
from eval.corpus_memoire import construire_corpus
from eval.golden import charger_golden

RACINE = Path(__file__).resolve().parent.parent
FICHIER_TRACES = RACINE / ".cache" / "eval_traces.jsonl"


def _trace(
    cas, resultats, reponse, verdict_local: str, brute=None, juges=None, generateurs=None,
    motif_fidelite=None, question=None,
) -> dict:
    """Un cas du golden dataset, prêt à être écrit en JSONL : les champs dont
    score_faithfulness.py a besoin pour rejouer la mesure indépendante.

    `generation` (réponse brute du LLM, citations comprises) et `articles`
    (articles récupérés) permettent de rejouer les garde-fous hors-ligne sur
    exactement la même réponse, sans nouvel appel Groq (rejouer.py)."""
    return {
        "id": cas.id,
        "question": cas.question,
        # Question de suivi : la question réécrite, celle qui a été cherchée.
        "question_comprise": question if question and question != cas.question else None,
        "reponse": reponse.reponse if reponse else "",
        "motif": reponse.motif if reponse else None,
        "contexts": _contextes_cites(reponse, resultats) if reponse and not reponse.refus else [],
        "verdict_local": verdict_local,
        # Motif du refus de l'étage 2 (affirmation rejetée) : `motif` est celui
        # de la réponse avant ce contrôle, vide pour une réponse rédigée.
        "motif_fidelite": motif_fidelite,
        "articles_attendus": [f"{cas.doc_id}:{a}" for a in cas.articles],
        "generation": brute.model_dump() if brute else None,
        # Modèle(s) qui ont jugé la fidélité (juge en cascade seulement).
        "modeles_juge": juges or [],
        # Modèle(s) qui ont rédigé la réponse : un relais (Mistral) n'écrit
        # pas comme Groq, le verdict en dépend.
        "modeles_generation": generateurs or [],
        "articles": [
            {"doc_id": r.doc_id, "content": r.content, "metadata": r.metadata}
            for r in resultats
        ],
    }


def evaluer(
    cas_golden, retriever, llm, nli, settings: Settings, k: int = 10, juge=None,
    juge_completude=None,
) -> Iterator[dict]:
    """Fait tourner recherche + génération + garde-fou étage 2 sur chaque cas,
    et produit une trace par cas (voir `_trace`) au fur et à mesure, pour
    qu'elle puisse être écrite avant de passer au cas suivant.

    `k` est le nombre d'articles récupérés, donc envoyés au LLM : moins
    d'articles consomme moins du quota Groq par réponse, au risque d'écarter
    l'article utile.

    Comme pipeline.repondre, une réponse fondée passe ensuite le contrôle de
    complétude (selon settings.controle_completude) ; une omission déclenche une
    seconde rédaction, dont la trace garde l'élément omis (`omission`).
    `juge_completude` remplace le juge pour ce seul contrôle (Kimi en
    analyse, cf. eval/kimi.py)."""
    for cas in cas_golden:
        # Comme pipeline.repondre : une question de suivi est d'abord rendue
        # autonome, puis suit le chemin habituel.
        question = reformuler_question(cas.question, cas.historique, llm)
        resultats = retriever.search(question, k=k)
        if not resultats:
            yield _trace(cas, resultats, None, "refus_retrieval", question=question)
            continue

        etat = _rediger(question, resultats, llm, nli, settings, juge)
        omission = None
        seconde_refusee = False
        if etat["verdict_local"] == "fondee" and _completude_a_controler(
            etat["reponse"], resultats, settings
        ):
            omission = verifier_completude(
                question, etat["reponse"], resultats, juge_completude or juge or llm
            )
            if omission:
                seconde = _rediger(question, resultats, llm, nli, settings, juge, omission)
                # Comme pipeline.repondre : une seconde rédaction refusée
                # laisse la place à la première, déjà vérifiée.
                seconde_refusee = seconde["verdict_local"] != "fondee"
                if not seconde_refusee:
                    etat = seconde
        trace = _trace(cas, resultats, question=question, **etat)
        trace["omission"] = omission
        trace["seconde_refusee"] = bool(omission) and seconde_refusee
        yield trace


def _rediger(question, resultats, llm, nli, settings: Settings, juge, omission=None) -> dict:
    """Rédaction puis garde-fous (citations, fidélité) : arguments de `_trace`
    décrivant le résultat, verdict compris."""
    historique_gen = getattr(llm, "historique", None)
    debut_gen = len(historique_gen) if historique_gen is not None else 0
    reponse_brute = generer_reponse(question, resultats, llm, omission)
    reponse_brute = corriger_citations(question, resultats, llm, reponse_brute)
    generateurs = historique_gen[debut_gen:] if historique_gen is not None else []
    etat = {"brute": reponse_brute, "generateurs": generateurs}
    if reponse_brute.refus:
        return {**etat, "reponse": reponse_brute, "verdict_local": "refus_generation"}

    reponse_citations = verifier_citations(reponse_brute, resultats)
    if reponse_citations.refus:
        return {**etat, "reponse": reponse_citations, "verdict_local": "refus_citation"}

    historique = getattr(juge or llm, "historique", None)
    debut = len(historique) if historique is not None else 0
    # reponse_citations est ce qu'on confronte à la mesure indépendante :
    # le texte tel que rédigé, avant que notre propre garde-fou étage 2 ne
    # le transforme éventuellement en refus.
    reponse_apres = verifier_fidelite(
        reponse_citations,
        resultats,
        nli,
        settings.seuil_nli_haut,
        settings.seuil_nli_contradiction,
        juge or llm,
    )
    return {
        **etat,
        "reponse": reponse_citations,
        "verdict_local": "refus_fidelite" if reponse_apres.refus else "fondee",
        "juges": historique[debut:] if historique is not None else [],
        "motif_fidelite": reponse_apres.motif if reponse_apres.refus else None,
    }


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
    # Défaut : le réglage de production (k_articles_generation). Jusqu'au
    # 2026-10-13, le défaut était 10, contre 5 en production : le golden
    # mesurait un autre système que celui qui répond aux utilisateurs.
    parser.add_argument(
        "--k", type=int, default=None,
        help="articles envoyés au LLM (défaut : k_articles_generation de la configuration)",
    )
    parser.add_argument(
        "--sortie", type=Path, default=FICHIER_TRACES, help="fichier de traces JSONL"
    )
    parser.add_argument("--ids", nargs="+", help="ne traiter que ces cas (ex. q22 q35)")
    parser.add_argument(
        "--juge-completude", choices=["groq", "kimi"], default="groq",
        help="juge du contrôle de complétude (kimi : analyse seulement, cf. eval/kimi.py)",
    )
    parser.add_argument(
        "--avec-hors-corpus",
        action="store_true",
        help="inclure les cas hors corpus (l'assistant doit refuser)",
    )
    args = parser.parse_args()
    fichier = args.sortie

    settings = Settings()
    embeddings = get_embedding_provider(settings)
    reranker = get_rerank_provider(settings)
    llm = get_llm(settings)
    juge = get_llm_juge(settings)
    nli = get_nli_provider(settings)
    juge_completude = None
    if args.juge_completude == "kimi":
        from eval.kimi import KimiLLM, lire_cle

        juge_completude = KimiLLM(lire_cle())

    print("Préparation du corpus en mémoire…")
    corpus = construire_corpus(embeddings)
    retriever = HybridRetriever(
        embeddings, corpus.dense, corpus.bm25, reranker=reranker,
        k_par_source=settings.k_par_source,
        k_avant_rerank=settings.k_avant_rerank,
        seuil_rerank=settings.seuil_garde_fou_retrieval,
        index_articles=corpus.bm25,
    )

    cas_golden = [c for c in charger_golden() if args.avec_hors_corpus or not c.hors_corpus]
    if args.ids:
        cas_golden = [c for c in cas_golden if c.id in args.ids]
    if args.limit:
        cas_golden = cas_golden[: args.limit]

    traces = traces_existantes(fichier) if args.reprendre else []
    deja_faits = {t["id"] for t in traces}
    a_faire = [c for c in cas_golden if c.id not in deja_faits]
    if deja_faits:
        print(f"Reprise : {len(deja_faits)} cas déjà tracés, conservés.")

    print(f"Génération de {len(a_faire)} traces (appels Groq + NLI réels)…")
    fichier.parent.mkdir(exist_ok=True)
    # Le fichier est réécrit avec les seules traces valides conservées, puis
    # chaque nouvelle trace y est ajoutée et vidée sur disque aussitôt.
    with fichier.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
        for i, t in enumerate(evaluer(a_faire, retriever, llm, nli, settings, k=args.k or settings.k_articles_generation, juge=juge,
            juge_completude=juge_completude), start=1):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            f.flush()
            traces.append(t)
            print(f"  [{i}/{len(a_faire)}] {t['id']} : {t['verdict_local']}")

    comptes: dict[str, int] = {}
    for t in traces:
        comptes[t["verdict_local"]] = comptes.get(t["verdict_local"], 0) + 1
    print(f"\nTraces écrites dans {fichier}")
    for verdict, n in sorted(comptes.items()):
        print(f"  {verdict:<18} {n}")
    print(
        "\nSuite : score_faithfulness.py (venv .venv-eval) pour la mesure "
        "indépendante et la comparaison."
    )


if __name__ == "__main__":
    main()
