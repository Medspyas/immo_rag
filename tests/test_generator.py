import json

import pytest

from app.config import Settings
from app.rag.generator import (
    Citation,
    GroqLLM,
    ReponseGeneree,
    generer_reponse,
    get_llm,
    verifier_citations,
)


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x", "groq_api_key": "fake-key"}
    base.update(overrides)
    return Settings(**base)


class _Resultat:
    """Forme minimale d'un article récupéré, attendue par generer_reponse."""

    def __init__(self, doc_id: str, content: str, article: str = "1", abroge: bool = False):
        self.doc_id = doc_id
        self.content = content
        self.metadata = {"article": article, "abroge": abroge}


class _LLMFake:
    """Renvoie une réponse figée et retient la consigne et le message reçus."""

    def __init__(self, sortie: str):
        self._sortie = sortie
        self.consigne_recue: str | None = None
        self.question_recue: str | None = None

    def generer(self, consigne: str, question: str) -> str:
        self.consigne_recue = consigne
        self.question_recue = question
        return self._sortie


_REPONSE_OK = json.dumps(
    {
        "refus": False,
        "reponse": "Le préavis du locataire est de trois mois.",
        "citations": [
            {"doc_id": "loi-89-462", "article": "15", "passage": "délai de préavis est de trois mois"}
        ],
    }
)


# --- get_llm -----------------------------------------------------------------


def test_groq_est_le_fournisseur_par_defaut():
    assert isinstance(get_llm(_settings()), GroqLLM)


def test_groq_sans_cle_api_leve_une_erreur():
    with pytest.raises(ValueError):
        get_llm(_settings(groq_api_key=None))


def test_fournisseur_inconnu_est_refuse():
    with pytest.raises(ValueError):
        _settings(llm_provider="mistral")


# --- generer_reponse -------------------------------------------------------


def test_sans_article_refuse_sans_appeler_le_modele():
    llm = _LLMFake(_REPONSE_OK)
    reponse = generer_reponse("Quel préavis ?", [], llm)
    assert reponse.refus is True
    assert llm.consigne_recue is None


def test_reponse_lue_depuis_le_json_du_modele():
    reponse = generer_reponse(
        "Quel préavis ?",
        [_Resultat("loi-89-462", "Le délai de préavis est de trois mois.", article="15")],
        _LLMFake(_REPONSE_OK),
    )
    assert reponse.refus is False
    assert "trois mois" in reponse.reponse
    assert reponse.citations[0].article == "15"


def test_refus_du_modele_est_transmis():
    sortie = json.dumps({"refus": True, "motif": "hors sujet", "reponse": "", "citations": []})
    reponse = generer_reponse("Météo demain ?", [_Resultat("loi-89-462", "Texte.")], _LLMFake(sortie))
    assert reponse.refus is True
    assert reponse.motif == "hors sujet"


def test_json_invalide_devient_un_refus():
    reponse = generer_reponse(
        "Quel préavis ?", [_Resultat("loi-89-462", "Texte.")], _LLMFake("pas du json")
    )
    assert reponse.refus is True
    assert "illisible" in reponse.motif


def test_json_sans_les_champs_attendus_devient_un_refus():
    reponse = generer_reponse(
        "Quel préavis ?", [_Resultat("loi-89-462", "Texte.")], _LLMFake('{"autre": 1}')
    )
    assert reponse.refus is True


def test_les_articles_et_la_question_sont_transmis_au_modele():
    llm = _LLMFake(_REPONSE_OK)
    generer_reponse(
        "Quel préavis ?",
        [_Resultat("loi-89-462", "Le délai de préavis est de trois mois.", article="15")],
        llm,
    )
    assert "Quel préavis ?" in llm.question_recue
    assert "loi-89-462" in llm.question_recue
    assert "Article 15" in llm.question_recue


def test_un_article_abroge_est_signale_au_modele():
    llm = _LLMFake(_REPONSE_OK)
    generer_reponse(
        "Question ?",
        [_Resultat("loi-x", "Ancien texte.", article="3", abroge=True)],
        llm,
    )
    assert "ABROGÉ" in llm.question_recue


def test_reponse_generee_valeurs_par_defaut():
    r = ReponseGeneree(refus=False)
    assert r.reponse == ""
    assert r.citations == []


# --- verifier_citations --------------------------------------------------

_ARTICLE_15 = _Resultat(
    "loi-89-462",
    "Lorsque le congé émane du locataire, le délai de préavis est de trois mois.",
    article="15",
)


def _reponse(citations, refus=False):
    return ReponseGeneree(
        refus=refus, reponse="Le préavis est de trois mois.", citations=citations
    )


def test_citation_presente_telle_quelle_est_conservee():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de trois mois")]
    )
    verifiee = verifier_citations(reponse, [_ARTICLE_15])
    assert verifiee.refus is False
    assert len(verifiee.citations) == 1


def test_citation_avec_ponctuation_et_casse_differentes_est_conservee():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="Le DÉLAI de préavis est de trois mois !")]
    )
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is False


def test_citation_inventee_fait_basculer_en_refus():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de six mois")]
    )
    verifiee = verifier_citations(reponse, [_ARTICLE_15])
    assert verifiee.refus is True
    assert "introuvable" in verifiee.motif


def test_une_seule_citation_fausse_suffit_a_refuser():
    reponse = _reponse(
        [
            Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de trois mois"),
            Citation(doc_id="loi-89-462", article="15", passage="le bailleur verse une indemnité"),
        ]
    )
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is True


def test_fragment_tres_court_exige_une_correspondance_exacte():
    reponse = _reponse([Citation(doc_id="loi-89-462", article="15", passage="trois mois")])
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is False


def test_passage_retrouve_dans_un_autre_article_recupere():
    # Le modèle annonce le mauvais numéro ; le passage existe dans un article récupéré.
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="99", passage="le délai de préavis est de trois mois")]
    )
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is False


def test_reponse_sans_citation_bascule_en_refus():
    verifiee = verifier_citations(_reponse([]), [_ARTICLE_15])
    assert verifiee.refus is True
    assert verifiee.motif == "réponse sans citation"


def test_un_refus_est_transmis_sans_verification():
    refus = ReponseGeneree(refus=True, motif="hors sujet")
    assert verifier_citations(refus, [_ARTICLE_15]) is refus
