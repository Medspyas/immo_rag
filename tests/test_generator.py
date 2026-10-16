import json
import re

import pytest

from app.config import Settings
from app.rag.generator import (
    GROQ_MAX_ESSAIS,
    LONGUEUR_MAX_CONTEXTE_CARACTERES,
    Citation,
    GroqLLM,
    LLMEnCascade,
    MistralLLM,
    ModeleIndisponible,
    QuotaJournalierEpuise,
    ReponseGeneree,
    _decouper_en_affirmations,
    _extraits,
    _preselectionner,
    corriger_citations,
    generer_reponse,
    get_llm,
    get_llm_juge,
    verifier_citations,
    verifier_fidelite,
)
from app.rag.nli import ScoreNLI


def _settings(**overrides) -> Settings:
    # Sans clé Mistral par défaut : celle du .env local ne doit pas changer
    # la cascade attendue par les tests.
    base = {"database_url": "postgresql://x", "groq_api_key": "fake-key", "mistral_api_key": None}
    base.update(overrides)
    return Settings(**base)


class _Resultat:
    """Forme minimale d'un article récupéré, attendue par generer_reponse."""

    def __init__(
        self, doc_id: str, content: str, article: str = "1", abroge: bool = False, section: str = ""
    ):
        self.doc_id = doc_id
        self.content = content
        self.metadata = {"article": article, "abroge": abroge, "section": section}


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
    assert isinstance(get_llm(_settings(llm_modeles_relais=[])), GroqLLM)


def test_client_groq_reessaie_sur_limite_de_debit():
    llm = get_llm(_settings(llm_modeles_relais=[]))
    assert llm._client.max_retries == GROQ_MAX_ESSAIS


def test_modeles_de_relais_donnent_une_cascade_sans_attente():
    llm = get_llm(_settings(llm_model="a", llm_modeles_relais=["b"]))
    assert isinstance(llm, LLMEnCascade)
    assert [m._model for m in llm._modeles] == ["a", "b"]
    # Dans la cascade, un modèle saturé passe la main au lieu d'attendre...
    assert all(m._client.max_retries == 0 for m in llm._modeles)
    # ...et seul le dernier recours attend la fin de la limite.
    assert [m._model for m in llm._derniers_recours] == ["a", "b"]
    assert all(m._client.max_retries == GROQ_MAX_ESSAIS for m in llm._derniers_recours)


def test_juge_place_son_modele_en_tete_de_cascade():
    juge = get_llm_juge(_settings(llm_model="a", llm_modeles_relais=["b"], llm_modele_juge="b"))
    assert [m._model for m in juge._modeles] == ["b", "a"]
    assert [m._model for m in juge._derniers_recours] == ["b", "a"]


def test_cle_mistral_ajoute_mistral_en_secours_seulement():
    llm = get_llm(_settings(llm_model="a", llm_modeles_relais=[], mistral_api_key="cle-m"))
    assert isinstance(llm, LLMEnCascade)
    # Hors de la cascade : une saturation d'une minute de Groq ne suffit pas.
    assert [type(m) for m in llm._modeles] == [GroqLLM]
    assert isinstance(llm._secours, MistralLLM)
    assert llm._secours._max_essais == GROQ_MAX_ESSAIS


def _mistral(gestionnaire, max_essais=0):
    import httpx

    attentes: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(gestionnaire))
    llm = MistralLLM("cle-m", "m", max_essais=max_essais, client=client, dormir=attentes.append)
    return llm, attentes


def test_mistral_renvoie_le_contenu_de_la_reponse():
    import httpx

    recues = []

    def _ok(requete):
        recues.append(requete)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"refus": true}'}}]})

    llm, _ = _mistral(_ok)
    assert llm.generer("consigne", "question") == '{"refus": true}'
    corps = json.loads(recues[0].content)
    assert corps["response_format"] == {"type": "json_object"}
    assert corps["messages"][0] == {"role": "system", "content": "consigne"}
    assert recues[0].headers["Authorization"] == "Bearer cle-m"


def test_mistral_impose_le_schema_selon_la_consigne():
    import httpx

    from app.rag.generator import CONSIGNE, CONSIGNE_JUGE_FIDELITE

    recues = []

    def _ok(requete):
        recues.append(json.loads(requete.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm, _ = _mistral(_ok)
    llm.generer(CONSIGNE, "q")
    llm.generer(CONSIGNE_JUGE_FIDELITE, "q")
    reponse, juge = (r["response_format"] for r in recues)
    assert reponse["type"] == juge["type"] == "json_schema"
    assert reponse["json_schema"]["schema"]["properties"]["reponse"] == {"type": "string"}
    assert juge["json_schema"]["schema"]["required"] == ["fondees"]


def test_mistral_sature_reessaie_apres_attente():
    import httpx

    reponses = iter(
        [
            httpx.Response(429, text="rate limit"),
            httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}),
        ]
    )
    llm, attentes = _mistral(lambda requete: next(reponses), max_essais=2)
    assert llm.generer("c", "q") == "{}"
    assert len(attentes) == 1


def test_mistral_sature_sans_essai_passe_la_main():
    import httpx

    llm, attentes = _mistral(lambda requete: httpx.Response(429, text="rate limit"))
    with pytest.raises(ModeleIndisponible):
        llm.generer("c", "q")
    assert attentes == []


class _LLMSature:
    def __init__(self):
        self.appels = 0

    def generer(self, consigne, question):
        self.appels += 1
        raise ModeleIndisponible("quota atteint")


class _LLMRepond:
    def __init__(self, texte):
        self.texte = texte
        self.appels = 0

    def generer(self, consigne, question):
        self.appels += 1
        return self.texte


def test_cascade_utilise_le_premier_modele_disponible():
    premier, second = _LLMRepond("premier"), _LLMRepond("second")
    cascade = LLMEnCascade([premier, second], derniers_recours=[_LLMRepond("r1"), _LLMRepond("r2")])
    assert cascade.generer("c", "q") == "premier"
    assert second.appels == 0


def test_cascade_passe_au_modele_suivant_quand_le_premier_est_sature():
    sature, second = _LLMSature(), _LLMRepond("second")
    cascade = LLMEnCascade([sature, second], derniers_recours=[_LLMRepond("r1"), _LLMRepond("r2")])
    assert cascade.generer("c", "q") == "second"
    assert sature.appels == 1


def test_cascade_note_le_modele_qui_a_repondu():
    sature, second = _LLMSature(), _LLMRepond("second")
    second.nom = "b"
    recours = _LLMRepond("recours")
    recours.nom = "a"
    cascade = LLMEnCascade([sature, second], derniers_recours=[recours, _LLMRepond("r2")])
    cascade.generer("c", "q")
    second.texte = None
    cascade._modeles = [_LLMSature(), _LLMSature()]
    cascade.generer("c", "q")
    assert cascade.historique == ["b", "a (attente)"]


def test_cascade_attend_via_le_dernier_recours_quand_tous_sont_satures():
    recours = _LLMRepond("recours")
    cascade = LLMEnCascade([_LLMSature(), _LLMSature()], derniers_recours=[recours, _LLMRepond("r2")])
    assert cascade.generer("c", "q") == "recours"
    assert recours.appels == 1


class _LLMEpuiseDuJour:
    def generer(self, consigne, question):
        raise QuotaJournalierEpuise("tokens per day")


def test_cascade_ne_attend_pas_un_modele_epuise_pour_la_journee():
    recours_epuise, recours_suivant = _LLMEpuiseDuJour(), _LLMRepond("second, patient")
    cascade = LLMEnCascade(
        [_LLMEpuiseDuJour(), _LLMSature()], derniers_recours=[recours_epuise, recours_suivant]
    )
    assert cascade.generer("c", "q") == "second, patient"
    assert recours_suivant.appels == 1


def test_cascade_tous_epuises_pour_la_journee_leve_quota_journalier():
    cascade = LLMEnCascade(
        [_LLMEpuiseDuJour(), _LLMEpuiseDuJour()],
        derniers_recours=[_LLMRepond("r1"), _LLMRepond("r2")],
    )
    with pytest.raises(QuotaJournalierEpuise):
        cascade.generer("c", "q")


def test_cascade_saturee_ou_epuisee_passe_au_secours():
    secours = _LLMRepond("secours")
    secours.nom = "m"
    cascade = LLMEnCascade(
        [_LLMSature(), _LLMEpuiseDuJour()],
        derniers_recours=[_LLMSature(), _LLMRepond("r2")],
        secours=secours,
    )
    assert cascade.generer("c", "q") == "secours"
    assert cascade.historique == ["m (secours)"]


def test_cascade_n_appelle_pas_le_secours_si_groq_repond_apres_attente():
    secours = _LLMRepond("secours")
    cascade = LLMEnCascade(
        [_LLMSature()], derniers_recours=[_LLMRepond("patient")], secours=secours
    )
    assert cascade.generer("c", "q") == "patient"
    assert secours.appels == 0


def test_cascade_ne_masque_pas_une_autre_erreur():
    class _LLMCasse:
        def generer(self, consigne, question):
            raise RuntimeError("bogue")

    cascade = LLMEnCascade([_LLMCasse(), _LLMRepond("second")], derniers_recours=[_LLMRepond("r"), _LLMRepond("r")])
    with pytest.raises(RuntimeError):
        cascade.generer("c", "q")


def test_groq_traduit_la_limite_de_debit_en_modele_indisponible(monkeypatch):
    import httpx
    from groq import RateLimitError

    llm = GroqLLM("fake-key", "a", max_essais=0)
    requete = httpx.Request("POST", "https://api.groq.com")
    erreur = RateLimitError(
        "quota", response=httpx.Response(429, request=requete), body=None
    )

    def _sature(consigne, question):
        raise erreur

    monkeypatch.setattr(llm, "_appeler", _sature)
    with pytest.raises(ModeleIndisponible):
        llm.generer("c", "q")


def test_groq_traduit_une_requete_trop_longue_en_modele_indisponible(monkeypatch):
    import httpx
    from groq import APIStatusError

    llm = GroqLLM("fake-key", "a", max_essais=0)
    requete = httpx.Request("POST", "https://api.groq.com")
    erreur = APIStatusError(
        "Request too large", response=httpx.Response(413, request=requete), body=None
    )

    def _trop_longue(consigne, question):
        raise erreur

    monkeypatch.setattr(llm, "_appeler", _trop_longue)
    with pytest.raises(ModeleIndisponible):
        llm.generer("c", "q")


def test_groq_traduit_un_json_invalide_en_modele_indisponible(monkeypatch):
    import httpx
    from groq import APIStatusError

    llm = GroqLLM("fake-key", "a", max_essais=0)
    requete = httpx.Request("POST", "https://api.groq.com")
    erreur = APIStatusError(
        "Error code: 400 - {'error': {'code': 'json_validate_failed'}}",
        response=httpx.Response(400, request=requete),
        body=None,
    )

    def _json_invalide(consigne, question):
        raise erreur

    monkeypatch.setattr(llm, "_appeler", _json_invalide)
    with pytest.raises(ModeleIndisponible):
        llm.generer("c", "q")


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


def test_la_section_de_larticle_est_transmise_au_modele():
    llm = _LLMFake(_REPONSE_OK)
    section = "Titre Ier bis : Des rapports entre bailleurs et locataires dans les logements meublés"
    generer_reponse(
        "Question ?",
        [_Resultat("loi-89-462", "Préavis d'un mois.", article="25-8", section=section)],
        llm,
    )
    assert f"Situé dans : {section}" in llm.question_recue


def test_le_bloc_darticles_sarrete_avant_de_depasser_le_budget():
    # Chaque article fait ~10 000 caractères : le 3e ferait dépasser le
    # budget de 24 000 (2 x ~10 030 = 20 060, + un 3e = 30 090) et doit donc
    # être omis, sans tronquer le 1er ni le 2e.
    llm = _LLMFake(_REPONSE_OK)
    articles = [_Resultat(f"doc-{i}", "x" * 10_000, article=str(i)) for i in range(1, 5)]
    generer_reponse("Question ?", articles, llm)
    assert "Article 1" in llm.question_recue
    assert "Article 2" in llm.question_recue
    assert "Article 3" not in llm.question_recue
    assert "Article 4" not in llm.question_recue


def test_le_premier_article_est_toujours_inclus_meme_hors_budget():
    llm = _LLMFake(_REPONSE_OK)
    articles = [_Resultat("loi-x", "x" * 30_000, article="1")]
    generer_reponse("Question ?", articles, llm)
    assert "Article 1" in llm.question_recue


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


def test_citation_abregee_par_points_de_suspension_est_conservee():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="Lorsque le congé … préavis est de trois mois.")]
    )
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is False


def test_citation_abregee_dans_le_desordre_fait_basculer_en_refus():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="est de trois mois ... Lorsque le congé")]
    )
    assert verifier_citations(reponse, [_ARTICLE_15]).refus is True


def test_citation_inventee_fait_basculer_en_refus():
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de six mois")]
    )
    verifiee = verifier_citations(reponse, [_ARTICLE_15])
    assert verifiee.refus is True
    assert "introuvable" in verifiee.motif


_ARTICLE_LONG = _Resultat(
    "loi-89-462",
    "Le délai de préavis est toutefois d'un mois : 1° Sur les territoires mentionnés au "
    "premier alinéa du I de l'article 17 ; 2° En cas d'obtention d'un premier emploi.",
    article="15",
)


def _citation_longue(passage):
    return _reponse([Citation(doc_id="loi-89-462", article="15", passage=passage)])


def test_faute_de_frappe_au_milieu_dune_citation_est_toleree():
    passage = (
        "toutefois d'un mois : 1° Sur les territoires mentionnés au premier "
        "alinéau du I de l'article 17 ; 2° En cas d'obtention d'un premier emploi"
    )
    assert verifier_citations(_citation_longue(passage), [_ARTICLE_LONG]).refus is False


def test_mot_remplace_au_milieu_dune_citation_est_refuse():
    passage = (
        "toutefois d'un mois : 1° Sur les communes mentionnées au premier "
        "alinéa du I de l'article 17 ; 2° En cas d'obtention d'un premier emploi"
    )
    assert verifier_citations(_citation_longue(passage), [_ARTICLE_LONG]).refus is True


def test_chiffre_modifie_au_milieu_dune_citation_est_refuse():
    passage = (
        "toutefois d'un mois : 1° Sur les territoires mentionnés au premier "
        "alinéau du I de l'article 18 ; 2° En cas d'obtention d'un premier emploi"
    )
    assert verifier_citations(_citation_longue(passage), [_ARTICLE_LONG]).refus is True


def test_citations_exactes_aucun_appel_de_correction():
    llm = _LLMFake(_REPONSE_OK)
    reponse = _reponse(
        [Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de trois mois")]
    )
    assert corriger_citations("Quel préavis ?", [_ARTICLE_15], llm, reponse) is reponse
    assert llm.question_recue is None


def test_citation_introuvable_donne_une_seconde_chance_au_modele():
    corrigee = json.dumps(
        {
            "refus": False,
            "reponse": "Le préavis est de trois mois.",
            "citations": [
                {"doc_id": "loi-89-462", "article": "15", "passage": "le délai de préavis est de trois mois"}
            ],
        }
    )
    llm = _LLMFake(corrigee)
    fautive = "Lorsque le congé vient du locataire, le préavis vaut trois mois"
    reponse = _reponse([Citation(doc_id="loi-89-462", article="15", passage=fautive)])
    nouvelle = corriger_citations("Quel préavis ?", [_ARTICLE_15], llm, reponse)
    assert fautive in llm.question_recue
    assert verifier_citations(nouvelle, [_ARTICLE_15]).refus is False


def test_seconde_chance_reste_dans_le_budget_de_caracteres():
    # Deux articles de ~11 000 caractères tiennent seuls dans le budget, mais
    # plus avec la réponse précédente : le second cède la place.
    articles = [
        _Resultat("loi-89-462", _ARTICLE_15.content + " " + "x" * 11_000, article="15"),
        _Resultat("loi-89-462", "y" * 11_000, article="16"),
    ]
    fautive = "Lorsque le congé vient du locataire, le préavis vaut trois mois " + "z" * 3_000
    reponse = _reponse([Citation(doc_id="loi-89-462", article="15", passage=fautive)])
    llm = _LLMFake(_REPONSE_OK)
    corriger_citations("Quel préavis ?", articles, llm, reponse)
    assert "Article 16" not in llm.question_recue
    assert len(llm.question_recue) <= LONGUEUR_MAX_CONTEXTE_CARACTERES + 1_000


def test_seconde_chance_toujours_fausse_reste_refusee():
    fautive = "Lorsque le congé vient du locataire, le préavis vaut trois mois"
    reponse = _reponse([Citation(doc_id="loi-89-462", article="15", passage=fautive)])
    llm = _LLMFake(reponse.model_dump_json())
    nouvelle = corriger_citations("Quel préavis ?", [_ARTICLE_15], llm, reponse)
    assert verifier_citations(nouvelle, [_ARTICLE_15]).refus is True


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
    """Renvoie des scores fixes quels que soient les textes reçus, et retient
    le dernier appel (prémisses, hypothèse) transmis."""

    def __init__(self, implication: float, contradiction: float = 0.0):
        self._score = ScoreNLI(implication=implication, contradiction=contradiction)
        self.dernier_appel: tuple[list[str], str] | None = None

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        self.dernier_appel = (premisses, hypothese)
        return [self._score for _ in premisses]


class _JugeLLM:
    """Juge de repli figé : renvoie toujours la même décision."""

    def __init__(self, fondee: bool):
        self._fondee = fondee
        self.appele = False
        self.appels = 0

    def generer(self, consigne: str, question: str) -> str:
        self.appele = True
        self.appels += 1
        self.question_recue = question
        liste = question.split("Affirmations à vérifier :")[-1]
        n = len(re.findall(r"^\d+\. ", liste, re.MULTILINE))
        return json.dumps({"fondees": [self._fondee] * n})


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
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_contradiction=0.7)
    assert verifiee.refus is False


def test_score_bas_sans_juge_bascule_en_refus():
    reponse = _reponse_avec_citation("Le préavis est de six mois.")
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.05), seuil_haut=0.7, seuil_contradiction=0.7)
    assert verifiee.refus is True
    assert "non fondée" in verifiee.motif


def test_score_bas_passe_aussi_au_juge():
    # Le NLI note bas des affirmations justes (régime tiré de l'intitulé,
    # abréviations) : un score bas n'est plus un refus direct.
    reponse = _reponse_avec_citation("En location vide, le préavis est de trois mois.")
    juge = _JugeLLM(fondee=True)
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.05), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert verifiee.refus is False
    assert juge.appele is True


def test_un_seul_appel_au_juge_pour_toutes_les_affirmations_douteuses():
    reponse = _reponse_avec_citation("Le préavis est court. Il est plutôt bref. Il dure peu.")
    juge = _JugeLLM(fondee=True)
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge)
    assert juge.appels == 1
    assert "3. Il dure peu." in juge.question_recue


class _JugeIncomplet:
    """Juge qui ne donne pas une décision par affirmation."""

    def generer(self, consigne: str, question: str) -> str:
        return json.dumps({"fondees": [True]})


def test_juge_incomplet_vaut_refus():
    reponse = _reponse_avec_citation("Le préavis est court. Il est plutôt bref.")
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=_JugeIncomplet()
    )
    assert verifiee.refus is True


def test_score_ambigu_sans_juge_bascule_en_refus_par_prudence():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7)
    assert verifiee.refus is True


def test_score_ambigu_le_juge_llm_peut_sauver_la_reponse():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    juge = _JugeLLM(fondee=True)
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert verifiee.refus is False
    assert juge.appele is True


def test_score_ambigu_le_juge_llm_peut_confirmer_le_refus():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=_JugeLLM(fondee=False)
    )
    assert verifiee.refus is True


def test_le_juge_recoit_tous_les_articles_cites_avec_leur_intitule():
    meuble = _Resultat(
        "loi-89-462", "Préavis d'un mois.", article="25-8", section="Titre Ier bis : logements meublés"
    )
    reponse = ReponseGeneree(
        refus=False,
        reponse="Trois mois en location vide, un mois en meublé.",
        citations=[
            Citation(doc_id="loi-89-462", article="15", passage="le délai de préavis est de trois mois"),
            Citation(doc_id="loi-89-462", article="25-8", passage="Préavis d'un mois"),
        ],
    )
    juge = _JugeLLM(fondee=True)
    verifier_fidelite(
        reponse, [_ARTICLE_15, meuble], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert "Article 15" in juge.question_recue
    assert "Article 25-8" in juge.question_recue
    assert "logements meublés" in juge.question_recue


def test_juge_llm_nest_pas_appele_hors_zone_ambigue():
    juge = _JugeLLM(fondee=False)
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge)
    assert juge.appele is False


def test_affirmation_fondee_sur_un_regime_particulier_part_au_juge():
    deux = _Resultat(
        "loi-65-557",
        "Le copropriétaire qui n'est pas syndic peut agir contre l'autre copropriétaire.",
        article="41-15",
        section="Section 2 : Dispositions particulières aux syndicats dont le nombre de voix est réparti entre deux copropriétaires",
    )
    reponse = ReponseGeneree(
        refus=False,
        reponse="Le copropriétaire qui n'est pas syndic peut agir contre l'autre copropriétaire.",
        citations=[Citation(doc_id="loi-65-557", article="41-15", passage="peut agir")],
    )
    juge = _JugeLLM(fondee=False)
    verifiee = verifier_fidelite(
        reponse, [deux], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert juge.appele is True
    assert verifiee.refus is True


class _NLISelonAffirmation:
    """Score haut pour les affirmations reprenant le texte de l'article, bas
    pour les autres — simule une affirmation fondée suivie d'une invention."""

    def evaluer(self, premisses: list[str], hypothese: str) -> list[ScoreNLI]:
        implication = 0.95 if "trois mois" in hypothese else 0.05
        return [ScoreNLI(implication=implication, contradiction=0.0) for _ in premisses]


def test_une_seule_affirmation_non_fondee_suffit_a_refuser():
    reponse = _reponse_avec_citation(
        "Le préavis est de trois mois. Le bailleur doit aussi verser une prime."
    )
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLISelonAffirmation(), seuil_haut=0.7, seuil_contradiction=0.7
    )
    assert verifiee.refus is True
    assert "prime" in verifiee.motif


def test_verifie_contre_le_contenu_de_larticle_cite():
    nli = _NLIFixe(0.95)
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifier_fidelite(reponse, [_ARTICLE_15], nli, seuil_haut=0.7, seuil_contradiction=0.7)
    assert nli.dernier_appel[0] == [_ARTICLE_15.content]


def test_un_refus_est_transmis_sans_verification_de_fidelite():
    refus = ReponseGeneree(refus=True, motif="hors sujet")
    assert verifier_fidelite(refus, [_ARTICLE_15], _NLIFixe(0.0), seuil_haut=0.7, seuil_contradiction=0.7) is refus


def test_reponse_vide_ne_leve_pas_derreur():
    reponse = ReponseGeneree(refus=False, reponse="", citations=[])
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.0), seuil_haut=0.7, seuil_contradiction=0.7)
    assert verifiee.refus is False


def test_contradiction_forte_empeche_lacceptation_directe():
    # Implication haute mais un extrait contredit : signaux opposés, sans juge
    # le garde-fou refuse par prudence.
    reponse = _reponse_avec_citation("Le préavis est de deux mois.")
    nli = _NLIFixe(0.95, contradiction=0.9)
    verifiee = verifier_fidelite(reponse, [_ARTICLE_15], nli, seuil_haut=0.7, seuil_contradiction=0.7)
    assert verifiee.refus is True


def test_contradiction_forte_passe_la_main_au_juge_llm():
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    juge = _JugeLLM(fondee=True)
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.95, contradiction=0.9),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.appele is True
    assert verifiee.refus is False


# --- découpage en affirmations, extraits, présélection ----------------------


def test_decoupe_une_affirmation_par_phrase():
    assert _decouper_en_affirmations("Le préavis est de trois mois. Il court dès réception.") == [
        "Le préavis est de trois mois.",
        "Il court dès réception.",
    ]


def test_elements_de_liste_rattaches_a_leur_introduction():
    texte = "Les missions du syndic sont :\n1. Conserver l'immeuble.\n2. Tenir la comptabilité."
    # « notamment » : un élément seul n'est qu'une partie de la liste ; sans
    # lui, le juge lisait « Les diagnostics sont un état du gaz » comme une
    # liste complète, donc fausse (q14, 2026-10-03).
    assert _decouper_en_affirmations(texte) == [
        "Les missions du syndic sont notamment : Conserver l'immeuble.",
        "Les missions du syndic sont notamment : Tenir la comptabilité.",
    ]


def test_liste_numerotee_sur_une_seule_ligne_decoupee_par_element():
    texte = "Les missions du syndic sont :\n1. Conserver l'immeuble. 2. Tenir la comptabilité."
    assert _decouper_en_affirmations(texte) == [
        "Les missions du syndic sont notamment : Conserver l'immeuble.",
        "Les missions du syndic sont notamment : Tenir la comptabilité.",
    ]


def test_reference_legale_ne_coupe_pas_la_phrase():
    texte = "Il faut :\n- le diagnostic prévu à l'article L. 126-26 du code. Il est annexé au bail selon l'art. 3."
    assert _decouper_en_affirmations(texte) == [
        "Il faut notamment : le diagnostic prévu à l'article L. 126-26 du code.",
        "Il est annexé au bail selon l'art. 3.",
    ]


def test_nombre_en_fin_de_phrase_nest_pas_un_element_de_liste():
    texte = "Le délai est fixé à l'article 3. Il court dès réception."
    assert _decouper_en_affirmations(texte) == [
        "Le délai est fixé à l'article 3.",
        "Il court dès réception.",
    ]


def test_liste_a_puces_sans_introduction_garde_les_elements_seuls():
    assert _decouper_en_affirmations("- Premier point.\n- Second point.") == [
        "Premier point.",
        "Second point.",
    ]


def test_une_phrase_ordinaire_clot_lintroduction_de_liste():
    texte = "Il faut :\n- une carte.\nLa carte est délivrée par la CCI.\n- Autre point."
    assert _decouper_en_affirmations(texte)[-1] == "Autre point."


def test_ligne_de_ponctuation_seule_nest_pas_une_affirmation():
    assert _decouper_en_affirmations("« Le préavis est de trois mois.\n»") == [
        "« Le préavis est de trois mois."
    ]


def test_oui_non_en_tete_est_retire():
    assert _decouper_en_affirmations("Oui, le préavis est de trois mois.") == [
        "le préavis est de trois mois."
    ]


def test_formule_dattribution_en_tete_est_retiree():
    texte = "Par ailleurs, l’article 19 précise que les créances sont garanties par une hypothèque."
    assert _decouper_en_affirmations(texte) == [
        "les créances sont garanties par une hypothèque."
    ]


def test_article_sujet_de_la_phrase_nest_pas_retire():
    texte = "L'article 19 concerne l'hypothèque légale."
    assert _decouper_en_affirmations(texte) == [texte]


def test_extraits_fenetre_glissante_de_trois_phrases():
    article = "Phrase un. Phrase deux. Phrase trois. Phrase quatre."
    assert _extraits([article]) == [
        "Phrase un. Phrase deux. Phrase trois.",
        "Phrase deux. Phrase trois. Phrase quatre.",
    ]


def test_extraits_article_court_donne_un_seul_extrait():
    assert _extraits(["Le délai de préavis est de trois mois."]) == [
        "Le délai de préavis est de trois mois."
    ]


def test_extraits_coupent_aux_alineas_et_points_virgules():
    assert _extraits(["Premier alinéa ;\nSecond alinéa"]) == ["Premier alinéa ; Second alinéa"]


def test_extraits_recollent_les_phrases_coupees_par_la_mise_en_page():
    # Retour à la ligne de PDF au milieu d'une phrase : la phrase reste entière.
    article = "Les conventions doivent être\nrédigées par écrit. Phrase deux.\nPhrase trois. Quatre."
    assert _extraits([article])[0] == (
        "Les conventions doivent être rédigées par écrit. Phrase deux. Phrase trois."
    )


def test_extraits_rattachent_les_elements_a_l_introduction():
    article = (
        "Les conventions doivent préciser :\nLes conditions de versement ;\n"
        "Les modalités de la reddition de compte ;\nLa rémunération ;\nLe mandant. Fin."
    )
    extraits = _extraits([article])
    # Un extrait commençant au milieu de l'énumération reprend l'introduction.
    assert extraits[3] == (
        "Les conventions doivent préciser : La rémunération ; Le mandant. Fin."
    )
    # L'énumération close par un point, l'introduction n'est plus reprise.
    assert _extraits(["Intro :\nUn ;\nDeux. Trois. Quatre. Cinq."])[-1] == "Trois. Quatre. Cinq."


def test_extraits_element_apres_un_point_interne_reprend_l_introduction():
    # « a) » contient un point : « c) » doit malgré tout garder l'introduction.
    article = (
        "Ne sont adoptées qu'à la majorité de tous les décisions concernant :\n"
        "a) La délégation au syndic. Elle fixe le montant ;\nb) La désignation du syndic ;\n"
        "c) Les travaux d'économies d'énergie."
    )
    assert _extraits([article])[-1] == (
        "Ne sont adoptées qu'à la majorité de tous les décisions concernant : "
        "a) La délégation au syndic. b) La désignation du syndic ; c) Les travaux d'économies d'énergie."
    )


def test_extraits_ajoutent_l_enumeration_complete():
    article = "Le dossier comprend :\n1° Le diagnostic ;\n2° Le constat ;\n3° L'état. Fin."
    assert _extraits([article])[-1] == (
        "Le dossier comprend : 1° Le diagnostic ; 2° Le constat ; 3° L'état."
    )


def test_preselection_garde_les_extraits_les_plus_proches_en_tete():
    extraits = ["Le bail est écrit.", "Le dépôt de garantie est d'un mois de loyer.", "Le syndic gère."]
    assert _preselectionner(extraits, "Le dépôt de garantie vaut un mois de loyer.")[0] == extraits[1]


def test_preselection_borne_le_nombre_dextraits():
    extraits = [f"Phrase numéro {i}." for i in range(50)]
    assert len(_preselectionner(extraits, "Phrase")) == 8


def test_numero_detape_et_renvoi_seul_ne_sont_pas_des_affirmations():
    texte = "Procédure prévue à l'article 20. 1. Le notaire avise le syndic. 3 bis. Le mandat, etc. (article 18-1 AA)."
    assert _decouper_en_affirmations(texte) == [
        "Procédure prévue à l'article 20.",
        "Le notaire avise le syndic.",
        "Le mandat, etc.",
    ]


def test_le_juge_recoit_la_reponse_complete_pour_les_renvois():
    texte = "Le préavis est de trois mois.\nCe délai est réduit à un mois en meublé."
    reponse = _reponse_avec_citation(texte)
    juge = _JugeLLM(fondee=True)
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge)
    recu = juge.question_recue
    assert texte in recu
    assert recu.index(texte) < recu.index("Affirmations à vérifier :")


def test_le_juge_recoit_les_passages_cites_en_tete():
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    juge = _JugeLLM(fondee=True)
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge)
    texte = juge.question_recue
    assert texte.index("« le délai de préavis est de trois mois »") < texte.index("Article 15\n")


def test_groq_distingue_le_quota_journalier(monkeypatch):
    import httpx
    from groq import RateLimitError

    llm = GroqLLM("fake-key", "a", max_essais=0)
    requete = httpx.Request("POST", "https://api.groq.com")
    erreur = RateLimitError(
        "Rate limit reached on tokens per day (TPD)",
        response=httpx.Response(429, request=requete),
        body=None,
    )

    def _epuise(consigne, question):
        raise erreur

    monkeypatch.setattr(llm, "_appeler", _epuise)
    with pytest.raises(QuotaJournalierEpuise):
        llm.generer("c", "q")


def test_sous_liste_dans_un_element_numerote():
    texte = (
        "Le locataire dispose de recours :\n"
        "1. Saisine du juge : le juge fixe les travaux. Le juge peut :\n"
        "   - réduire le loyer ;\n"
        "   - suspendre le bail.\n"
        "2. Saisine de la commission : elle rend un avis."
    )
    affirmations = _decouper_en_affirmations(texte)
    assert "Le juge peut" not in affirmations
    assert "Le juge peut notamment : réduire le loyer ;" in affirmations
    assert "Le locataire dispose de recours notamment : Saisine de la commission : elle rend un avis." in affirmations


def test_sous_liste_en_lettres():
    # q03 (2026-10-14) : « a. » sortait comme une affirmation à part entière
    texte = (
        "Le préavis est d'un mois :\n"
        "1. Pour un bail classique, si l'une des conditions est remplie :\n"
        "   a. Le locataire obtient un premier emploi.\n"
        "   b) Le locataire perd son emploi."
    )
    affirmations = _decouper_en_affirmations(texte)
    assert "a." not in affirmations
    assert (
        "Pour un bail classique, si l'une des conditions est remplie notamment : "
        "Le locataire obtient un premier emploi." in affirmations
    )
    assert any(a.endswith("Le locataire perd son emploi.") for a in affirmations)


def test_decoupage_retire_la_mise_en_forme_markdown():
    texte = (
        "## Règle\n"
        "**Règles générales :**\n"
        "- Le préavis est de *trois mois*.\n"
        "* Il court dès **réception** de la lettre."
    )
    assert _decouper_en_affirmations(texte) == [
        "Règles générales notamment : Le préavis est de trois mois.",
        "Règles générales notamment : Il court dès réception de la lettre.",
    ]
