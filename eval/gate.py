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
- part des questions hors corpus arrêtées par le garde-fou étage 1 (meilleur
  score de reclassement sous seuil_garde_fou_retrieval). Sur le corpus complet
  (code civil et code des impôts compris, 2026-10-13), aucun seuil ne sépare
  plus les deux groupes : une question hors corpus proche d'un article réel
  (durée de validité du DPE, 0,83) dépasse de bonnes questions (IFI, 0,04).
  L'étage 1 n'arrête donc qu'une partie des questions hors corpus ; les autres
  relèvent de l'étage 2, mesuré de bout en bout par run_eval.py. Depuis le
  2026-10-27 (seuil 0,1, cf. config.py), l'étage 1 n'arrête plus aucune
  question hors corpus du golden, par choix : la part bloquée et le meilleur
  score (hors_corpus_max) restent affichés, pour information, sans plancher.

Les planchers sont relevés à partir d'une mesure de référence, avec une petite
marge : le reclasseur ne donne pas exactement les mêmes scores sur carte
graphique (ONNX) et sur processeur.

Lancement :
    python -m eval.gate                  # mesure et contrôle
    python -m eval.gate --threads 2      # limite l'usage CPU
    python -m eval.gate --fichier eval/questions_neuves.jsonl
        # mesure sur un autre jeu de questions, détail par question,
        # sans contrôle des planchers (réglés sur le golden)
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


def intervalle(rangs: list[int | None], mesure: str, tirages: int = 2000, graine: int = 0) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une mesure (« mrr » ou « hit@k »),
    par rééchantillonnage des questions (bootstrap).

    Sur 40 questions, une seule question fait bouger hit@k de 2,5 points :
    sans marge d'erreur, un écart entre deux mesures ne dit pas s'il vient du
    système ou du hasard des questions tirées. Le bootstrap retire au hasard,
    avec remise, autant de questions que le jeu en compte, recalcule la mesure,
    et garde les 2,5 % et 97,5 % des valeurs obtenues."""
    import random

    if not rangs:
        return (0.0, 0.0)
    if mesure == "mrr":
        valeurs = [1 / r if r else 0.0 for r in rangs]
    else:
        k = int(mesure.split("@")[1])
        valeurs = [1.0 if r and r <= k else 0.0 for r in rangs]
    hasard = random.Random(graine)
    n = len(valeurs)
    moyennes = sorted(sum(hasard.choices(valeurs, k=n)) / n for _ in range(tirages))
    return (moyennes[int(0.025 * tirages)], moyennes[int(0.975 * tirages) - 1])


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


def mesurer(settings, chemin: Path | None = None, detail: bool = False) -> dict[str, float]:
    """Mesures du gate sur le golden dataset (ou le fichier chemin),
    configuration de production. detail affiche le rang de chaque question."""
    from app.rag.embeddings import get_embedding_provider
    from app.rag.reranker import get_rerank_provider
    from app.rag.retriever import HybridRetriever
    from eval.ablation import KS, evaluer_systeme, recherche_hybride
    from eval.corpus_memoire import construire_corpus
    from eval.golden import charger_golden, questions_autonomes

    corpus = construire_corpus(get_embedding_provider(settings))
    reranker = get_rerank_provider(settings)
    tous = questions_autonomes(charger_golden(chemin))
    dans = [c for c in tous if not c.hors_corpus]
    hors = [c for c in tous if c.hors_corpus]

    resultat = evaluer_systeme(
        "hybride+rerank",
        recherche_hybride(
            corpus,
            reranker=reranker,
            k_avant_rerank=settings.k_avant_rerank,
            k_par_source=settings.k_par_source,
            rerank_fenetres=settings.rerank_fenetres,
            rerank_meilleure_fenetre=settings.rerank_meilleure_fenetre,
        ),
        dans,
    )
    if detail:
        for c, rang in zip(dans, resultat.rangs, strict=True):
            print(f"{c.id:<5} rang {rang or 'absent':<7} {c.question}")
    mesures = {f"hit@{k}": resultat.hit_at[k] for k in KS}
    mesures["mrr"] = resultat.mrr
    if detail:
        print(f"\nIntervalles de confiance à 95 % ({len(dans)} questions) :")
        for nom in ("hit@1", "hit@5", "mrr"):
            bas, haut = intervalle(resultat.rangs, nom)
            print(f"  {nom:<6} {mesures[nom]:.3f}  [{bas:.3f} ; {haut:.3f}]")

    retriever = HybridRetriever(
        corpus.embeddings,
        corpus.dense,
        corpus.bm25,
        reranker=reranker,
        k_par_source=settings.k_par_source,
        k_avant_rerank=settings.k_avant_rerank,
        index_articles=corpus.bm25,
        rerank_fenetres=settings.rerank_fenetres,
        rerank_meilleure_fenetre=settings.rerank_meilleure_fenetre,
    )
    scores_hors = []
    for c in hors:
        top = retriever.search(c.question, 1)
        scores_hors.append((top[0].score_rerank or 0.0) if top else 0.0)
        if detail:
            print(f"{c.id:<5} hors corpus, meilleur score {scores_hors[-1]:.3f}  {c.question}")
    mesures["hors_corpus_max"] = max(scores_hors, default=0.0)
    seuil = settings.seuil_garde_fou_retrieval
    mesures["hors_corpus_bloques"] = (
        sum(s < seuil for s in scores_hors) / len(scores_hors) if scores_hors else 1.0
    )
    return mesures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="cœurs CPU max (défaut 4)")
    parser.add_argument("--seuils", type=Path, default=SEUILS, help="fichier des planchers")
    parser.add_argument(
        "--fichier", type=Path, default=None,
        help="autre jeu de questions (même format) : détail affiché, planchers non contrôlés",
    )
    args = parser.parse_args()

    import torch

    from app.config import Settings

    torch.set_num_threads(max(1, args.threads))
    settings = Settings()
    seuils = json.loads(args.seuils.read_text(encoding="utf-8"))

    mesures = mesurer(settings, args.fichier, detail=args.fichier is not None)
    for nom, valeur in mesures.items():
        print(f"{nom:<20} {valeur:.3f}")

    if args.fichier is not None:
        return
    echecs = verifier_seuils(mesures, seuils)
    if echecs:
        print("\nGate en échec :")
        for e in echecs:
            print(f"  - {e}")
        sys.exit(1)
    print("\nGate franchi.")


if __name__ == "__main__":
    main()
