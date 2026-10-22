"""Mesure du temps de réponse, étape par étape : recherche seule ou réponse
complète.

Avant d'accélérer la recherche, il faut savoir où passe le temps. Une requête
enchaîne quatre étapes (cf. retriever.py) :
  1. vecteur de la question  (modèle d'embeddings) ;
  2. recherche par le sens   (produit scalaire, ici en mémoire) ;
  3. recherche par mots-clés (BM25) ;
  4. reclassement            (cross-encoder BGE-m3, le plus lourd a priori).

Avec --complet, la mesure couvre la réponse entière (pipeline.repondre) :
la recherche ci-dessus, puis la génération par le LLM (Groq), la
vérification de fidélité par le modèle NLI et, sur les cas ambigus, le juge
LLM. Appels Groq réels : limiter le nombre de questions (--questions).

Chaque étape est enveloppée dans un chronomètre, puis les questions du golden
dataset passent par le vrai code du produit (HybridRetriever, repondre), sans
copie de sa logique. Le corpus est celui de l'ablation (en mémoire, sans base
de données).

Pour chaque étape : médiane et p95 (95 % des requêtes vont plus vite que cette
valeur), comptés par requête — une étape absente d'une requête (juge LLM non
sollicité, génération évitée par le garde-fou étage 1) y compte pour 0. La
colonne « requêtes » indique combien de requêtes l'ont réellement exécutée.
Le total est le temps de bout en bout de la requête. La première question, qui
charge les modèles en mémoire, est écartée du calcul : ce coût n'est payé
qu'une fois, au démarrage du serveur.

Lancement :
    python -m eval.latence                      # reclasseur réglé dans .env
    python -m eval.latence --reclasseur local   # force le reclasseur CPU
    python -m eval.latence --reclasseur onnx    # force le reclasseur ONNX
    python -m eval.latence --sans-reclasseur    # recherche hybride seule
    python -m eval.latence --reclasseur local --k-avant-rerank 10 --max-length 256 --questions 8
                                                # reclasseur allégé, 8 questions
    python -m eval.latence --complet --questions 10   # réponse complète (Groq réel)
"""
from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Même bridage CPU que l'ablation, avant tout import de NumPy/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time

from app.config import Settings
from app.rag.embeddings import get_embedding_provider
from app.rag.retriever import HybridRetriever
from eval.corpus_memoire import construire_corpus
from eval.golden import charger_golden, questions_autonomes

ETAPES = (
    "vecteur question",
    "sens",
    "mots-clés",
    "reclassement",
    "génération",
    "NLI",
    "juge LLM",
)
TOTAL = "total"


class Chronometre:
    """Relève la durée (en millisecondes) des appels à une méthode d'un objet,
    sans changer son résultat.

    Les mesures sont rangées par requête : `nouvelle_requete` ouvre une case,
    et chaque appel y ajoute sa durée sous le nom de son étape (plusieurs
    appels d'une même étape s'additionnent)."""

    def __init__(self):
        self.requetes: list[dict[str, float]] = []

    def nouvelle_requete(self) -> None:
        self.requetes.append({})

    def ajouter(self, etape: str, duree_ms: float) -> None:
        requete = self.requetes[-1]
        requete[etape] = requete.get(etape, 0.0) + duree_ms

    def envelopper(self, objet, methode: str, etape: str, etape_suivante: str | None = None):
        """Renvoie un objet dont `methode` est chronométrée ; le reste des
        attributs est celui de l'objet d'origine.

        etape_suivante : nom donné aux appels suivants au sein d'une même
        requête. Le LLM sert d'abord à rédiger la réponse, puis, le cas
        échéant, de juge : même méthode, deux étapes distinctes."""
        chrono = self
        originale = getattr(objet, methode)

        class _Enveloppe:
            def __getattr__(self, nom):
                return getattr(objet, nom)

        def mesuree(*args, **kwargs):
            nom = etape
            if etape_suivante is not None and etape in chrono.requetes[-1]:
                nom = etape_suivante
            debut = time.perf_counter()
            resultat = originale(*args, **kwargs)
            chrono.ajouter(nom, (time.perf_counter() - debut) * 1000)
            return resultat

        enveloppe = _Enveloppe()
        setattr(enveloppe, methode, mesuree)
        return enveloppe


def percentile(valeurs: list[float], p: float) -> float:
    """Valeur sous laquelle se trouvent p % des mesures (p entre 0 et 100),
    par interpolation linéaire entre les deux mesures voisines."""
    if not valeurs:
        raise ValueError("aucune mesure")
    tries = sorted(valeurs)
    position = (len(tries) - 1) * p / 100
    bas = int(position)
    haut = min(bas + 1, len(tries) - 1)
    return tries[bas] + (tries[haut] - tries[bas]) * (position - bas)


def _ligne(nom: str, mesures: list[float], executees: int) -> str:
    return (
        f"{nom:<18}{percentile(mesures, 50):>8.0f}ms{percentile(mesures, 95):>8.0f}ms"
        f"{executees:>8}/{len(mesures)}"
    )


def tableau(requetes: list[dict[str, float]], ignorer_premiere: bool = True) -> str:
    """Tableau médiane / p95 par étape, puis le total de bout en bout.

    Sans mesure « total », le total est la somme des étapes de la requête."""
    retenues = requetes[1:] if ignorer_premiere else requetes
    if not retenues:
        raise ValueError("aucune requête mesurée")
    largeur = 50
    lignes = [f"{'étape':<18}{'médiane':>10}{'p95':>10}{'requêtes':>12}", "-" * largeur]
    for etape in ETAPES:
        executees = sum(1 for r in retenues if etape in r)
        if executees:
            lignes.append(_ligne(etape, [r.get(etape, 0.0) for r in retenues], executees))
    totaux = [r.get(TOTAL, sum(v for e, v in r.items() if e != TOTAL)) for r in retenues]
    lignes.append("-" * largeur)
    lignes.append(_ligne(TOTAL, totaux, len(retenues)))
    return "\n".join(lignes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    groupe = parser.add_mutually_exclusive_group()
    groupe.add_argument("--reclasseur", choices=["local", "onnx"], help="force le reclasseur")
    groupe.add_argument("--sans-reclasseur", action="store_true", help="hybride seule")
    parser.add_argument("--threads", type=int, default=4, help="cœurs CPU max (défaut 4)")
    parser.add_argument(
        "--k-avant-rerank", type=int, default=20, help="passages soumis au reclasseur (défaut 20)"
    )
    parser.add_argument(
        "--max-length", type=int, help="longueur max. en tokens par paire (défaut : .env)"
    )
    parser.add_argument(
        "--questions", type=int, help="n'utilise que les N premières questions du golden"
    )
    parser.add_argument(
        "--complet", action="store_true", help="réponse complète : génération + garde-fous (Groq)"
    )
    args = parser.parse_args()
    if args.complet and args.sans_reclasseur:
        parser.error("--complet mesure le produit réel, qui utilise le reclasseur")

    import torch

    torch.set_num_threads(max(1, args.threads))

    settings = Settings()
    reglages = {}
    if args.reclasseur:
        reglages["rerank_provider"] = args.reclasseur
    if args.max_length:
        reglages["rerank_max_length"] = args.max_length
    settings = settings.model_copy(update=reglages)

    corpus = construire_corpus(get_embedding_provider(settings))
    chrono = Chronometre()

    reranker = None
    if not args.sans_reclasseur:
        from app.rag.reranker import get_rerank_provider

        reranker = chrono.envelopper(get_rerank_provider(settings), "rerank", "reclassement")

    retriever = HybridRetriever(
        chrono.envelopper(corpus.embeddings, "embed_query", "vecteur question"),
        chrono.envelopper(corpus.dense, "search", "sens"),
        chrono.envelopper(corpus.bm25, "search", "mots-clés"),
        reranker=reranker,
        k_avant_rerank=args.k_avant_rerank,
        # Le garde-fou étage 1 n'intervient que dans la réponse complète : la
        # recherche seule est mesurée sans refus, comme dans l'ablation.
        seuil_rerank=settings.seuil_garde_fou_retrieval if args.complet else None,
        index_articles=corpus.bm25,
    )

    if args.complet:
        from app.rag.generator import get_llm
        from app.rag.nli import get_nli_provider
        from app.rag.pipeline import repondre

        llm = chrono.envelopper(get_llm(settings), "generer", "génération", "juge LLM")
        nli = chrono.envelopper(get_nli_provider(settings), "evaluer", "NLI")

        def executer(question: str) -> None:
            repondre(question, retriever, llm, nli, settings)
    else:

        def executer(question: str) -> None:
            retriever.search(question, k=10)

    questions = [c.question for c in questions_autonomes(charger_golden())][: args.questions]
    for i, question in enumerate(questions, start=1):
        chrono.nouvelle_requete()
        debut = time.perf_counter()
        executer(question)
        chrono.ajouter(TOTAL, (time.perf_counter() - debut) * 1000)
        print(f"  {i}/{len(questions)}", end="\r", flush=True)

    nom = "aucun" if args.sans_reclasseur else settings.rerank_provider
    if not args.sans_reclasseur:
        nom += f", {args.k_avant_rerank} passages, {settings.rerank_max_length} tokens max."
    quoi = "la réponse complète" if args.complet else "la recherche"
    print(f"Latence de {quoi} — {len(questions)} questions, reclasseur : {nom}")
    print(f"({len(corpus.chunks)} articles, 1re question écartée : chargement des modèles)\n")
    print(tableau(chrono.requetes))


if __name__ == "__main__":
    main()
