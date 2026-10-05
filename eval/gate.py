"""Gate d'évaluation : bloque une modification qui dégrade le retrieval.

Une modification du découpage, des embeddings ou du reclasseur peut faire
reculer la recherche sans casser aucun test unitaire : les tests vérifient la
logique, pas la qualité. Ce script mesure la qualité sur le golden dataset,
avec la configuration de production (recherche hybride + reclasseur), et la
compare à des planchers fixés dans eval/seuils_gate.json. Un seul plancher
franchi suffit à faire échouer le script (code de sortie 1), donc la CI.

Mesures contrôlées :
- hit@k et MRR sur les questions du corpus (cf. metrics.py) : le bon article
  remonte-t-il, et assez haut ?
- meilleur score de reclassement sur les questions hors corpus : il doit rester
  sous le seuil du garde-fou étage 1 (seuil_garde_fou_retrieval), sinon une
  question sans réponse dans la base passerait jusqu'au LLM.

Les planchers sont relevés à partir d'une mesure de référence, avec une petite
marge : le reclasseur ne donne pas exactement les mêmes scores sur carte
graphique (ONNX) et sur processeur.

Lancement :
    python -m eval.gate                  # mesure et contrôle
    python -m eval.gate --threads 2      # limite l'usage CPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Même bridage que ablation.py, avant tout import de NumPy/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

SEUILS = Path(__file__).with_name("seuils_gate.json")


def verifier_seuils(mesures: dict[str, float], seuils: dict[str, dict[str, float]]) -> list[str]:
    """Liste des planchers franchis, vide si tout passe.

    seuils associe à chaque mesure soit {"min": x} (la mesure doit valoir au
    moins x), soit {"max": x} (au plus x). Une mesure attendue mais absente
    compte comme un échec : un gate qui ne contrôle plus rien doit se voir."""
    echecs: list[str] = []
    for nom, borne in seuils.items():
        if nom not in mesures:
            echecs.append(f"{nom} : mesure absente")
            continue
        valeur = mesures[nom]
        if "min" in borne and valeur < borne["min"]:
            echecs.append(f"{nom} = {valeur:.3f} < plancher {borne['min']:.3f}")
        if "max" in borne and valeur > borne["max"]:
            echecs.append(f"{nom} = {valeur:.3f} > plafond {borne['max']:.3f}")
    return echecs


def mesurer(settings) -> dict[str, float]:
    """Mesures du gate sur le golden dataset, configuration de production."""
    from app.rag.embeddings import get_embedding_provider
    from app.rag.reranker import get_rerank_provider
    from app.rag.retriever import HybridRetriever
    from eval.ablation import KS, evaluer_systeme, recherche_hybride
    from eval.corpus_memoire import construire_corpus
    from eval.golden import charger_golden

    corpus = construire_corpus(get_embedding_provider(settings))
    reranker = get_rerank_provider(settings)
    tous = charger_golden()
    dans = [c for c in tous if not c.hors_corpus]
    hors = [c for c in tous if c.hors_corpus]

    resultat = evaluer_systeme(
        "hybride+rerank",
        recherche_hybride(
            corpus,
            reranker=reranker,
            k_avant_rerank=settings.k_avant_rerank,
            k_par_source=settings.k_par_source,
        ),
        dans,
    )
    mesures = {f"hit@{k}": resultat.hit_at[k] for k in KS}
    mesures["mrr"] = resultat.mrr

    retriever = HybridRetriever(
        corpus.embeddings,
        corpus.dense,
        corpus.bm25,
        reranker=reranker,
        k_par_source=settings.k_par_source,
        k_avant_rerank=settings.k_avant_rerank,
    )
    scores_hors = []
    for c in hors:
        top = retriever.search(c.question, 1)
        scores_hors.append((top[0].score_rerank or 0.0) if top else 0.0)
    mesures["hors_corpus_max"] = max(scores_hors, default=0.0)
    return mesures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="cœurs CPU max (défaut 4)")
    parser.add_argument("--seuils", type=Path, default=SEUILS, help="fichier des planchers")
    args = parser.parse_args()

    import torch

    from app.config import Settings

    torch.set_num_threads(max(1, args.threads))
    settings = Settings()
    seuils = json.loads(args.seuils.read_text(encoding="utf-8"))
    # Le plafond hors corpus suit le seuil du garde-fou étage 1 s'il n'est pas
    # fixé à la main : c'est lui qui décide du refus en production.
    seuils.setdefault("hors_corpus_max", {"max": settings.seuil_garde_fou_retrieval})

    mesures = mesurer(settings)
    for nom, valeur in mesures.items():
        print(f"{nom:<16} {valeur:.3f}")

    echecs = verifier_seuils(mesures, seuils)
    if echecs:
        print("\nGate en échec :")
        for e in echecs:
            print(f"  - {e}")
        sys.exit(1)
    print("\nGate franchi.")


if __name__ == "__main__":
    main()
