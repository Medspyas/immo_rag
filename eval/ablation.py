"""Étude d'ablation du retrieval : la recherche hybride vaut-elle sa complexité ?

Le checkpoint de CLAUDE.md : avant d'empiler d'autres briques sur la recherche,
mesurer si l'hybride (sens + mots-clés + fusion RRF) fait vraiment mieux que la
recherche par le sens seule, sur le golden dataset. Si l'écart est faible, on
coupe la complexité qui ne se justifie pas.

Chaque question du golden est posée aux systèmes comparés ; on regarde à quel
rang le premier article attendu remonte, puis on agrège en hit@k et MRR
(cf. metrics.py).

Systèmes comparés :
  - « sens seul »   : recherche par le sens uniquement (dense) ;
  - « hybride »      : dense + BM25 + fusion RRF ;
  - « hybride+rerank » : idem, plus le reclasseur BGE-m3 (option --rerank ;
                         lent sur CPU, désactivé par défaut).

Lancement :
    python -m eval.ablation                 # sens seul vs hybride
    python -m eval.ablation --rerank         # ajoute le reclasseur
    python -m eval.ablation --contexte       # situe chaque article dans son
                                              # document avant l'embedding
                                              # (contextual retrieval, Groq)
    python -m eval.ablation --threads 2      # limite l'usage CPU
    python -m eval.ablation --rerank --k-avant-rerank 10 --max-length 256
                                              # reclasseur allégé (cf.
                                              # eval/latence.py pour la vitesse)

Le premier lancement calcule les embeddings du corpus (~1-2 min, un pic CPU
borné par --threads) puis les met en cache ; les suivants sont immédiats.
"""
from __future__ import annotations

import argparse
import os
import sys

# La sortie contient des accents ; sur une console Windows en page de code
# héritée, forcer UTF-8 évite les caractères illisibles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Bride le calcul parallèle AVANT tout import qui chargerait NumPy/PyTorch : ces
# bibliothèques lisent le nombre de threads à l'import. --threads l'ajuste
# ensuite pour PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.rag.embeddings import get_embedding_provider
from app.rag.retriever import HybridRetriever
from eval.corpus_memoire import CorpusMemoire, construire_corpus
from eval.golden import CasGolden, charger_golden, questions_autonomes
from eval.metrics import Source, mrr, rang_premier_pertinent, taux_hit_at_k

# Rangs examinés. On récupère toujours K_MAX résultats ; hit@k se lit pour
# k <= K_MAX, le MRR porte sur l'ensemble.
KS = (1, 3, 5, 10)
K_MAX = 20

Recherche = Callable[[str, int], list[Source]]


@dataclass
class ResultatEval:
    nom: str
    n_questions: int
    hit_at: dict[int, float]
    mrr: float
    rangs: list[int | None]


def evaluer_systeme(nom: str, rechercher: Recherche, cas: list[CasGolden]) -> ResultatEval:
    """Pose chaque question au système, relève le rang du premier article
    attendu, agrège en hit@k et MRR."""
    rangs: list[int | None] = []
    for c in cas:
        resultats = rechercher(c.question, K_MAX)
        cibles = {(c.doc_id, article) for article in c.articles}
        rangs.append(rang_premier_pertinent(resultats, cibles))
    return ResultatEval(
        nom=nom,
        n_questions=len(cas),
        hit_at={k: taux_hit_at_k(rangs, k) for k in KS},
        mrr=mrr(rangs),
        rangs=rangs,
    )


def _sources(resultats) -> list[Source]:
    return [(r.doc_id, r.metadata.get("article", "")) for r in resultats]


def recherche_dense(corpus: CorpusMemoire) -> Recherche:
    def rechercher(question: str, k: int) -> list[Source]:
        vecteur = corpus.embeddings.embed_query(question)
        return _sources(corpus.dense.search(vecteur, k))

    return rechercher


def recherche_hybride(
    corpus: CorpusMemoire,
    *,
    reranker=None,
    k_avant_rerank: int = 20,
    k_par_source: int = 20,
) -> Recherche:
    retriever = HybridRetriever(
        corpus.embeddings,
        corpus.dense,
        corpus.bm25,
        reranker=reranker,
        k_par_source=k_par_source,
        k_avant_rerank=k_avant_rerank,
        index_articles=corpus.bm25,
    )

    def rechercher(question: str, k: int) -> list[Source]:
        return _sources(retriever.search(question, k))

    return rechercher


def _tableau(resultats: list[ResultatEval]) -> str:
    lignes = [f"{'système':<18} " + "  ".join(f"hit@{k:<3}" for k in KS) + "   MRR"]
    lignes.append("-" * len(lignes[0]))
    for r in resultats:
        cases = "  ".join(f"{r.hit_at[k]:6.1%}" for k in KS)
        lignes.append(f"{r.nom:<18} {cases}   {r.mrr:.3f}")
    return "\n".join(lignes)


def _details(resultats: list[ResultatEval], cas: list[CasGolden]) -> str:
    """Rang du premier article attendu, question par question, pour chaque
    système. « — » quand l'article n'est pas dans les 20 premiers résultats."""
    entetes = [r.nom for r in resultats]
    lignes = [f"{'id':<5} {'  '.join(f'{n:>14}' for n in entetes)}   question"]
    for i, c in enumerate(cas):
        rangs = "  ".join(
            f"{(str(r.rangs[i]) if r.rangs[i] is not None else '—'):>14}"
            for r in resultats
        )
        lignes.append(f"{c.id:<5} {rangs}   {c.question[:60]}")
    return "\n".join(lignes)


def _resume(xs: list[float]) -> str:
    if not xs:
        return "—"
    return f"min {xs[0]:.3f} | médiane {xs[len(xs) // 2]:.3f} | max {xs[-1]:.3f}"


def _diagnostic_hors_corpus(
    corpus: CorpusMemoire,
    cas: list[CasGolden],
    *,
    reranker=None,
    k_avant_rerank: int = 20,
    k_par_source: int = 20,
) -> str:
    """Compare, pour les questions dans le corpus vs hors corpus, le meilleur
    score du passage n°1. Un écart net entre les deux groupes indiquerait qu'un
    seuil sur ce score peut servir de garde-fou (refus) — l'étage 1.

    Le score de sens (cosinus) est toujours affiché. Quand un reclasseur est
    fourni, on ajoute le meilleur score de reclassement (0–1) : c'est lui que
    le garde-fou étage 1 doit porter d'après le plan, la comparaison sert à en
    fixer le seuil.
    """
    def meilleur_cosinus(question: str) -> float:
        vecteur = corpus.embeddings.embed_query(question)
        top = corpus.dense.search(vecteur, 1)
        return top[0].score if top else 0.0

    dans = sorted(meilleur_cosinus(c.question) for c in cas if not c.hors_corpus)
    hors = sorted(meilleur_cosinus(c.question) for c in cas if c.hors_corpus)

    lignes = [
        "Meilleur score de sens (cosinus) :",
        f"  questions DANS le corpus  ({len(dans):>2}) : {_resume(dans)}",
        f"  questions HORS corpus     ({len(hors):>2}) : {_resume(hors)}",
    ]

    if reranker is not None:
        retriever = HybridRetriever(
            corpus.embeddings,
            corpus.dense,
            corpus.bm25,
            reranker=reranker,
            k_par_source=k_par_source,
            k_avant_rerank=k_avant_rerank,
            index_articles=corpus.bm25,
        )

        def meilleur_rerank(question: str) -> float:
            top = retriever.search(question, 1)
            return top[0].score_rerank if top and top[0].score_rerank is not None else 0.0

        dans_r = sorted(meilleur_rerank(c.question) for c in cas if not c.hors_corpus)
        hors_r = sorted(meilleur_rerank(c.question) for c in cas if c.hors_corpus)
        lignes += [
            "",
            "Meilleur score de reclassement (0–1) :",
            f"  questions DANS le corpus  ({len(dans_r):>2}) : {_resume(dans_r)}",
            f"  questions HORS corpus     ({len(hors_r):>2}) : {_resume(hors_r)}",
        ]

    return "\n".join(lignes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerank", action="store_true", help="ajoute le reclasseur BGE-m3")
    parser.add_argument(
        "--contexte", action="store_true", help="active le contextual retrieval (Groq)"
    )
    parser.add_argument("--threads", type=int, default=4, help="cœurs CPU max (défaut 4)")
    parser.add_argument("--details", action="store_true", help="rang par question")
    parser.add_argument(
        "--reclasseur", choices=["local", "onnx"], help="force le reclasseur (défaut : .env)"
    )
    parser.add_argument(
        "--k-avant-rerank", type=int, help="passages soumis au reclasseur (défaut : .env)"
    )
    parser.add_argument(
        "--k-par-source", type=int,
        help="résultats demandés à chaque recherche avant fusion (défaut : .env)",
    )
    parser.add_argument(
        "--max-length", type=int, help="longueur max. en tokens par paire (défaut : .env)"
    )
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))

    settings = Settings()
    reglages = {}
    if args.reclasseur:
        reglages["rerank_provider"] = args.reclasseur
    if args.max_length:
        reglages["rerank_max_length"] = args.max_length
    if args.k_avant_rerank:
        reglages["k_avant_rerank"] = args.k_avant_rerank
    if args.k_par_source:
        reglages["k_par_source"] = args.k_par_source
    settings = settings.model_copy(update=reglages)
    embeddings = get_embedding_provider(settings)

    contextualizer = None
    if args.contexte:
        from app.rag.contextualizer import get_context_generator

        contextualizer = get_context_generator(settings)

    print("Préparation du corpus en mémoire…")
    corpus = construire_corpus(embeddings, contextualizer=contextualizer)
    print(f"{len(corpus.chunks)} articles indexés" + (" (contextualisés).\n" if args.contexte else ".\n"))

    tous = questions_autonomes(charger_golden())
    dans_corpus = [c for c in tous if not c.hors_corpus]

    systemes: list[tuple[str, Recherche]] = [
        ("sens seul", recherche_dense(corpus)),
        ("hybride", recherche_hybride(corpus, k_par_source=settings.k_par_source)),
    ]
    reranker = None
    if args.rerank:
        from app.rag.reranker import get_rerank_provider

        reranker = get_rerank_provider(settings)
        systemes.append(
            (
                "hybride+rerank",
                recherche_hybride(
                    corpus,
                    reranker=reranker,
                    k_avant_rerank=settings.k_avant_rerank,
                    k_par_source=settings.k_par_source,
                ),
            )
        )
        print(
            f"Reclasseur : {settings.rerank_provider}, {settings.k_avant_rerank} passages, "
            f"{settings.rerank_max_length} tokens max.\n"
        )

    resultats = [evaluer_systeme(nom, fn, dans_corpus) for nom, fn in systemes]

    print(f"Ablation retrieval — {len(dans_corpus)} questions du corpus\n")
    print(_tableau(resultats))
    print()
    print(
        _diagnostic_hors_corpus(
            corpus,
            tous,
            reranker=reranker,
            k_avant_rerank=settings.k_avant_rerank,
            k_par_source=settings.k_par_source,
        )
    )
    if args.details:
        print()
        print(_details(resultats, dans_corpus))


if __name__ == "__main__":
    main()
