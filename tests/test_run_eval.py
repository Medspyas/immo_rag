"""Tests de la phase 1 de l'évaluation faithfulness (eval/run_eval.py).

Aucun appel réseau : LLM, NLI et retriever sont des doublures. On vérifie le
câblage (quel verdict_local pour quel scénario, quel contenu dans la trace),
pas la qualité d'un vrai modèle.
"""
from __future__ import annotations

import json
import re

from app.config import Settings
from app.rag.generator import CONSIGNE, CONSIGNE_JUGE_FIDELITE
from app.rag.nli import ScoreNLI
from app.rag.suivi import CONSIGNE_SUIVI, Echange
from eval.golden import CasGolden
from eval.run_eval import evaluer, traces_existantes


class _Resultat:
    def __init__(self, doc_id: str, content: str, article: str = "1"):
        self.doc_id = doc_id
        self.content = content
        self.metadata = {"article": article, "abroge": False}


class _RetrieverFake:
    """Renvoie une liste de résultats fixe, ou vide selon la question."""

    def __init__(self, resultats: list[_Resultat]):
        self._resultats = resultats

    def search(self, question: str, k: int) -> list[_Resultat]:
        return [] if question == "hors corpus" else self._resultats


class _LLMFake:
    """Distingue l'appel de génération de l'appel du juge de fidélité par la
    consigne reçue, comme le fait le vrai GroqLLM avec deux prompts différents."""

    def __init__(self, sortie_generation: str, fondee: bool = True):
        self._sortie_generation = sortie_generation
        self._fondee = fondee

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_JUGE_FIDELITE:
            liste = question.split("Affirmations à vérifier :")[-1]
            return json.dumps({"fondees": [self._fondee] * len(re.findall(r"^\d+\. ", liste, re.MULTILINE))})
        assert consigne == CONSIGNE
        return self._sortie_generation


class _NLIFake:
    """Score d'implication fixe, ambigu par défaut (force le repli sur le juge
    LLM dans les tests qui l'exercent)."""

    def __init__(self, score: float = 0.5):
        self._score = score

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        return [ScoreNLI(implication=self._score, contradiction=0.0) for _ in premisses]


RESULTATS = [_Resultat("loi-89-462", "Le délai de préavis est de trois mois.", "15")]

_CAS_TROUVE = CasGolden(
    id="c1", question="préavis du locataire", categorie="x",
    doc_id="loi-89-462", articles=["15"], reponse_type="reponse",
)
_CAS_HORS_CORPUS = CasGolden(
    id="c2", question="hors corpus", categorie="x",
    doc_id=None, articles=[], reponse_type="refus", hors_corpus=True,
)

_REPONSE_OK = json.dumps({
    "refus": False,
    "reponse": "Le préavis du locataire est de trois mois.",
    "citations": [{"doc_id": "loi-89-462", "article": "15", "passage": "délai de préavis est de trois mois"}],
})
_REPONSE_REFUS = json.dumps({"refus": True, "motif": "articles insuffisants", "reponse": "", "citations": []})
_REPONSE_CITATION_INVENTEE = json.dumps({
    "refus": False,
    "reponse": "Le préavis est de six mois.",
    "citations": [{"doc_id": "loi-89-462", "article": "15", "passage": "passage qui n'existe pas dans la source"}],
})


def _settings() -> Settings:
    return Settings(database_url="postgresql://x", groq_api_key="fake-key")


def test_verdict_refus_retrieval_si_aucun_resultat():
    traces = list(evaluer(
        [_CAS_HORS_CORPUS], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_OK), _NLIFake(), _settings()
    ))
    assert traces[0]["verdict_local"] == "refus_retrieval"
    assert traces[0]["contexts"] == []


def test_verdict_refus_generation_si_le_llm_refuse():
    traces = list(evaluer(
        [_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_REFUS), _NLIFake(), _settings()
    ))
    assert traces[0]["verdict_local"] == "refus_generation"


def test_verdict_refus_citation_si_la_citation_est_introuvable():
    traces = list(evaluer(
        [_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_CITATION_INVENTEE), _NLIFake(), _settings()
    ))
    assert traces[0]["verdict_local"] == "refus_citation"


def test_trace_garde_la_generation_brute_et_les_articles():
    # Même refusée, la réponse brute du LLM est conservée avec ses citations,
    # pour pouvoir rejouer les garde-fous hors-ligne.
    traces = list(evaluer(
        [_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_CITATION_INVENTEE), _NLIFake(), _settings()
    ))
    generation = traces[0]["generation"]
    assert generation["citations"][0]["passage"] == "passage qui n'existe pas dans la source"
    assert traces[0]["articles"] == [
        {"doc_id": "loi-89-462", "content": "Le délai de préavis est de trois mois.", "metadata": {"article": "15", "abroge": False}}
    ]


def test_verdict_fondee_quand_nli_haut_confirme():
    traces = list(evaluer(
        [_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_OK), _NLIFake(score=0.9), _settings()
    ))
    assert traces[0]["verdict_local"] == "fondee"
    assert traces[0]["reponse"] == "Le préavis du locataire est de trois mois."
    assert traces[0]["contexts"] == ["Le délai de préavis est de trois mois."]
    assert traces[0]["articles_attendus"] == ["loi-89-462:15"]


def test_verdict_refus_fidelite_quand_le_juge_llm_dit_non_fondee():
    traces = list(evaluer(
        [_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_OK, fondee=False), _NLIFake(score=0.5),
        _settings(),
    ))
    assert traces[0]["verdict_local"] == "refus_fidelite"
    assert traces[0]["motif_fidelite"]
    # La réponse comparée à la mesure indépendante reste celle AVANT le refus
    # du garde-fou : c'est elle que score_faithfulness.py doit noter.
    assert traces[0]["reponse"] == "Le préavis du locataire est de trois mois."


# --- traces_existantes (reprise après interruption) ---------------------------


def test_traces_existantes_fichier_absent(tmp_path):
    assert traces_existantes(tmp_path / "absent.jsonl") == []


def test_traces_existantes_ecarte_une_ligne_coupee(tmp_path):
    fichier = tmp_path / "traces.jsonl"
    # Deuxième ligne coupée net, comme après une interruption en pleine écriture.
    complete = json.dumps({"id": "q01", "verdict_local": "fondee"})
    fichier.write_text(complete + "\n" + '{"id": "q02", "verd', encoding="utf-8")
    assert [t["id"] for t in traces_existantes(fichier)] == ["q01"]



def test_evaluer_transmet_k_a_la_recherche():
    class _RetrieverEspion(_RetrieverFake):
        def search(self, question: str, k: int) -> list[_Resultat]:
            self.k = k
            return super().search(question, k)

    retriever = _RetrieverEspion(RESULTATS)
    list(evaluer([_CAS_TROUVE], retriever, _LLMFake(_REPONSE_OK), _NLIFake(score=0.9), _settings(), k=5))
    assert retriever.k == 5


def test_question_de_suivi_reecrite_avant_la_recherche():
    class _RetrieverEspion(_RetrieverFake):
        def search(self, question: str, k: int) -> list[_Resultat]:
            self.question = question
            return super().search(question, k)

    class _LLMSuivi(_LLMFake):
        def generer(self, consigne: str, question: str) -> str:
            if consigne == CONSIGNE_SUIVI:
                return json.dumps({"question": "Quel est le préavis du locataire ?"})
            return super().generer(consigne, question)

    cas = CasGolden(
        id="c3", question="et pour le locataire ?", categorie="x",
        doc_id="loi-89-462", articles=["15"], reponse_type="reponse",
        historique=[Echange(question="préavis du bailleur", reponse="Six mois.")],
    )
    retriever = _RetrieverEspion(RESULTATS)
    [trace] = evaluer([cas], retriever, _LLMSuivi(_REPONSE_OK), _NLIFake(score=0.9), _settings())
    assert retriever.question == "Quel est le préavis du locataire ?"
    assert trace["question"] == "et pour le locataire ?"
    assert trace["question_comprise"] == "Quel est le préavis du locataire ?"


def test_question_autonome_sans_question_comprise():
    [trace] = evaluer([_CAS_TROUVE], _RetrieverFake(RESULTATS), _LLMFake(_REPONSE_OK), _NLIFake(score=0.9), _settings())
    assert trace["question_comprise"] is None
