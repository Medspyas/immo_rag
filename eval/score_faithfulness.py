"""Phase 2 de l'évaluation faithfulness : la mesure indépendante, et sa
comparaison au garde-fou en ligne.

Relit les traces produites par run_eval.py (recherche + génération + garde-
fou étage 2 déjà faits, dans le venv habituel du projet) et calcule, pour
chaque cas, la métrique `faithfulness` de deepeval : un juge LLM (Groq)
décompose la réponse en affirmations puis vérifie chacune contre le contexte
récupéré — la même idée que notre `verifier_fidelite`, mais indépendante (pas
notre propre code, pas notre propre modèle NLI) et plus coûteuse (plusieurs
appels LLM par cas, pas un petit modèle local).

Ce module vit dans un venv séparé (`.venv-eval`, cf. CLAUDE.md section
Commandes) : deepeval entraîne des dépendances (client OpenAI, etc.) qui
n'ont rien à faire dans l'image de prod ni dans le venv de l'app, et qui
pourraient entrer en conflit avec les versions dont `generator.py` a besoin.

Lancement (depuis .venv-eval) :
    python -m eval.score_faithfulness                  # tous les cas de la trace
    python -m eval.score_faithfulness --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

RACINE = Path(__file__).resolve().parent.parent
FICHIER_TRACES = RACINE / ".cache" / "eval_traces.jsonl"

# Verdicts de run_eval.py qui correspondent à une réponse notée par
# verifier_fidelite (donc comparable à la mesure indépendante). Les refus
# d'étages antérieurs (retrieval, citation) sont un désaccord différent —
# hors du périmètre de la métrique faithfulness, qui juge la fidélité d'un
# TEXTE RÉDIGÉ, pas l'absence de réponse.
VERDICTS_COMPARABLES = {"fondee", "refus_fidelite"}


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """Fait jouer à Groq le rôle de juge indépendant pour deepeval.

    Même réglage que GroqLLM (app/rag/generator.py) : le modèle par défaut
    (openai/gpt-oss-20b) est un modèle de raisonnement — sans borner ce
    raisonnement, il épuise son budget de tokens avant d'écrire la réponse
    attendue (constaté en conditions réelles le 2026-09-18, même bug corrigé
    à deux endroits déjà). deepeval retente sans schéma structuré si
    `generate` ne l'accepte pas (cf. DeepEvalBaseLLM.generate_with_schema) :
    cette implémentation reste volontairement minimale, texte en - texte out.
    """

    def load_model(self):
        from groq import Groq

        # Plafond de tokens par minute du forfait gratuit : réessayer après
        # l'attente indiquée par Groq (valeur alignée sur generator.py, non
        # importée : ce venv n'installe pas le paquet du produit).
        return Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6)

    def generate(self, prompt: str) -> str:
        reponse = self.model.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # deepeval demande en interne des sorties structurées plus
            # longues que notre propre juge (liste d'affirmations extraites
            # du contexte, puis liste de verdicts un par un) : avec 1024,
            # constaté en conditions réelles (2026-09-22) que le JSON est
            # coupé en plein milieu d'une chaîne (~1200 tokens de sortie),
            # ce que deepeval ne peut pas parser. Relevé à 4096.
            max_tokens=4096,
            reasoning_effort="low",
        )
        return reponse.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.name


def charger_traces(fichier: Path) -> list[dict]:
    with fichier.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def comparer(traces: list[dict], metric: FaithfulnessMetric) -> list[dict]:
    """Calcule le score deepeval pour chaque trace comparable et le confronte
    au verdict de notre garde-fou en ligne."""
    resultats = []
    for t in traces:
        if t["verdict_local"] not in VERDICTS_COMPARABLES:
            continue
        cas_test = LLMTestCase(
            input=t["question"], actual_output=t["reponse"], retrieval_context=t["contexts"]
        )
        metric.measure(cas_test)
        verdict_deepeval = "fondee" if metric.success else "refus_fidelite"
        resultats.append(
            {
                "id": t["id"],
                "verdict_local": t["verdict_local"],
                "score_deepeval": metric.score,
                "verdict_deepeval": verdict_deepeval,
                "accord": t["verdict_local"] == verdict_deepeval,
            }
        )
    return resultats


def _tableau(resultats: list[dict]) -> str:
    lignes = [f"{'id':<5} {'garde-fou local':<16} {'deepeval':<10} {'score':<7} accord"]
    lignes.append("-" * len(lignes[0]))
    for r in resultats:
        lignes.append(
            f"{r['id']:<5} {r['verdict_local']:<16} {r['verdict_deepeval']:<10} "
            f"{r['score_deepeval']:<7.2f} {'oui' if r['accord'] else 'NON'}"
        )
    return "\n".join(lignes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="limiter aux N premiers cas comparables")
    parser.add_argument("--threshold", type=float, default=0.5, help="seuil de succès deepeval (défaut 0.5)")
    args = parser.parse_args()

    if "GROQ_API_KEY" not in os.environ:
        try:
            from dotenv import load_dotenv

            load_dotenv(RACINE / ".env")
        except ImportError:
            pass
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("GROQ_API_KEY manquant (ni dans l'environnement, ni dans .env).")

    if not FICHIER_TRACES.exists():
        raise SystemExit(
            f"{FICHIER_TRACES} introuvable — lancer d'abord run_eval.py (venv .venv, cf. CLAUDE.md)."
        )

    traces = charger_traces(FICHIER_TRACES)
    comparables = [t for t in traces if t["verdict_local"] in VERDICTS_COMPARABLES]
    if args.limit:
        comparables = comparables[: args.limit]

    # openai/gpt-oss-20b (modèle par défaut de la production, generator.py)
    # génère parfois un JSON mal formé sur les sorties longues de deepeval
    # (jusqu'à ~60 affirmations extraites d'un article long) : constaté en
    # conditions réelles (2026-09-22) — pas une troncature (max_tokens déjà
    # monté à 4096), le modèle oublie simplement de fermer le tableau JSON.
    # gpt-oss-120b, plus grand mais toujours gratuit sur Groq, ne présente
    # pas ce défaut sur les mêmes cas. Ce choix ne concerne que ce juge
    # indépendant : la production continue d'utiliser le modèle par défaut.
    llm = GroqDeepEvalLLM(model="openai/gpt-oss-120b")
    metric = FaithfulnessMetric(threshold=args.threshold, model=llm, include_reason=False, async_mode=False)

    print(f"Notation deepeval de {len(comparables)} cas comparables (appels Groq indépendants)…")
    resultats = comparer(comparables, metric)

    print()
    print(_tableau(resultats))
    accord = sum(r["accord"] for r in resultats)
    print(f"\nAccord garde-fou local / deepeval : {accord}/{len(resultats)}")


if __name__ == "__main__":
    main()
