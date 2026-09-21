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
    verifier_fidelite,
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


# --- verifier_fidelite (garde-fou étage 2) ----------------------------------


class _NLIFixe:
    """Renvoie un score fixe quels que soient les textes reçus, et retient le
    dernier couple (prémisse, hypothèse) transmis."""

    def __init__(self, score: float):
        self._score = score
        self.dernier_appel: tuple[str, str] | None = None

    def score_implication(self, premisse: str, hypothese: str) -> float:
        self.dernier_appel = (premisse, hypothese)
        return self._score


class _JugeLLM:
    """Juge de repli figé : renvoie toujours la même décision."""

    def __init__(self, fondee: bool):
        self._sortie = json.dumps({"fondee": fondee})
        self.appele = False

    def generer(self, consigne: str, question: str) -> str:
        self.appele = True
        return self._sortie


def _reponse_avec_citation(texte: str, article: str = "15") -> ReponseGeneree:
    return ReponseGeneree(
        refus=False,
        reponse=texte,
        citations=[
            Citation(doc_id="loi-89-462", article=article, passage="le délai de préavis est de trois mois")
        ],
    )


def test_score_au_dessus_du_seuil_haut_est_conserve():
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_bas=0.3)
    assert verifiee.refus is False


def test_score_sous_le_seuil_bas_bascule_en_refus_sans_juge():
    reponse = _reponse_avec_citation("Le préavis est de six mois.")
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.05), seuil_haut=0.7, seuil_bas=0.3)
    assert verifiee.refus is True
    assert "non fondée" in verifiee.motif


def test_score_ambigu_sans_juge_bascule_en_refus_par_prudence():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_bas=0.3)
    assert verifiee.refus is True


def test_score_ambigu_le_juge_llm_peut_sauver_la_reponse():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    juge = _JugeLLM(fondee=True)
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_bas=0.3, llm=juge
    )
    assert verifiee.refus is False
    assert juge.appele is True


def test_score_ambigu_le_juge_llm_peut_confirmer_le_refus():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_bas=0.3, llm=_JugeLLM(fondee=False)
    )
    assert verifiee.refus is True


def test_juge_llm_nest_pas_appele_hors_zone_ambigue():
    juge = _JugeLLM(fondee=False)
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_bas=0.3, llm=juge)
    assert juge.appele is False


class _NLISelonAffirmation:
    """Score haut pour les affirmations reprenant le texte de l'article, bas
    pour les autres — simule une affirmation fondée suivie d'une invention."""

    def score_implication(self, premisse: str, hypothese: str) -> float:
        return 0.95 if "trois mois" in hypothese else 0.05


def test_une_seule_affirmation_non_fondee_suffit_a_refuser():
    reponse = _reponse_avec_citation(
        "Le préavis est de trois mois. Le bailleur doit aussi verser une prime."
    )
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLISelonAffirmation(), seuil_haut=0.7, seuil_bas=0.3
    )
    assert verifiee.refus is True
    assert "prime" in verifiee.motif


def test_verifie_contre_le_contenu_de_larticle_cite():
    nli = _NLIFixe(0.95)
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifier_fidelite(reponse, [_ARTICLE_15], nli, seuil_haut=0.7, seuil_bas=0.3)
    assert nli.dernier_appel[0] == _ARTICLE_15.content


def test_un_refus_est_transmis_sans_verification_de_fidelite():
    refus = ReponseGeneree(refus=True, motif="hors sujet")
    assert verifier_fidelite(refus, [_ARTICLE_15], _NLIFixe(0.0), seuil_haut=0.7, seuil_bas=0.3) is refus


def test_reponse_vide_ne_leve_pas_derreur():
    reponse = ReponseGeneree(refus=False, reponse="", citations=[])
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.0), seuil_haut=0.7, seuil_bas=0.3)
    assert verifiee.refus is False
