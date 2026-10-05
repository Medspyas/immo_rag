"""Tests du rejeu hors-ligne des garde-fous (eval/rejouer.py).

Aucun appel réseau : le NLI est une doublure à score fixe.
"""
from __future__ import annotations

from app.config import Settings
from app.rag.nli import ScoreNLI
from eval.rejouer import rejouer_trace

_ARTICLES = [
    {"doc_id": "loi-89-462", "content": "Le délai de préavis est de trois mois.", "metadata": {"article": "15"}}
]


class _NLIFake:
    def __init__(self, score: float):
        self._score = score

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        return [ScoreNLI(implication=self._score, contradiction=0.0) for _ in premisses]


def _settings() -> Settings:
    return Settings(database_url="postgresql://x", groq_api_key="fake-key")


def _trace(passage: str) -> dict:
    return {
        "id": "c1",
        "verdict_local": "fondee",
        "generation": {
            "refus": False,
            "reponse": "Le préavis est de trois mois.",
            "citations": [{"doc_id": "loi-89-462", "article": "15", "passage": passage}],
        },
        "articles": _ARTICLES,
    }


def test_rejeu_fondee_quand_le_nli_confirme():
    verdict, _ = rejouer_trace(_trace("délai de préavis est de trois mois"), _NLIFake(0.99), _settings())
    assert verdict == "fondee"


def test_rejeu_a_juger_quand_le_nli_hesite():
    verdict, _ = rejouer_trace(_trace("délai de préavis est de trois mois"), _NLIFake(0.5), _settings())
    assert verdict == "a_juger"


def test_rejeu_refus_citation_quand_le_passage_est_invente():
    verdict, motif = rejouer_trace(_trace("passage qui n'existe pas du tout"), _NLIFake(0.99), _settings())
    assert verdict == "refus_citation"
    assert "introuvable" in motif


def test_rejeu_garde_le_verdict_sans_generation():
    trace = {"id": "c2", "verdict_local": "refus_retrieval", "motif": None, "generation": None, "articles": []}
    assert rejouer_trace(trace, _NLIFake(0.99), _settings()) == ("refus_retrieval", None)
