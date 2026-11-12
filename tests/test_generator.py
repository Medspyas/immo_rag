import json
import re

import pytest

from app.config import Settings
from app.rag.generator import (
    CONSIGNE_JUGE_EXCEPTION,
    CONSIGNE_JUGE_FIDELITE,
    CONSIGNE_JUGE_REPONSE_DIRECTE,
    CONSIGNE_OUI_NON,
    CONSIGNE_ROLES,
    CONSIGNE_ROLES_ARTICLE,
    CONSIGNE_ROLES_QUESTION,
    CONSIGNE_SITUATION_EXCEPTION,
    GROQ_MAX_ESSAIS,
    LONGUEUR_MAX_CONTEXTE_CARACTERES,
    Citation,
    GroqLLM,
    LLMEnCascade,
    ModeleIndisponible,
    QuotaJournalierEpuise,
    ReponseGeneree,
    _bloc_articles,
    _decouper_en_affirmations,
    _exonerations_derogeant,
    _extraits,
    _preselectionner,
    _retirer_affirmations,
    champ_application,
    completude_a_verifier,
    corriger_citations,
    generer_reponse,
    get_llm,
    get_llm_juge,
    get_llm_roles,
    oui_non_a_relire,
    rappeler_champ,
    reparer_citations,
    retirer_exemples,
    roles_a_verifier,
    sans_exemples,
    verifier_citations,
    verifier_completude,
    verifier_fidelite,
    verifier_roles,
)
from app.rag.nli import ScoreNLI


def _settings(**overrides) -> Settings:
    base = {"database_url": "postgresql://x", "groq_api_key": "fake-key"}
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


def test_redacteur_des_roles_en_tete_de_cascade_ou_absent():
    llm = get_llm_roles(_settings(llm_model="a", llm_modeles_relais=["b"], llm_modele_roles="q"))
    assert [m._model for m in llm._modeles] == ["q", "a", "b"]
    assert get_llm_roles(_settings(llm_modele_roles="")) is None


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


def test_cascade_saturee_partout_echoue_sans_secours():
    # Plus de relais hors Groq (Mistral retiré) : l'erreur remonte, l'API
    # répond « service saturé » plutôt qu'une réponse dégradée.
    cascade = LLMEnCascade(
        [_LLMSature(), _LLMEpuiseDuJour()],
        derniers_recours=[_LLMSature(), _LLMRepond("r2")],
    )
    with pytest.raises(ModeleIndisponible):
        cascade.generer("c", "q")


def test_cascade_dernier_recours_en_echec_passe_au_suivant():
    # x024 (2026-11-11) : qwen, dernier recours, renvoie un JSON invalide ;
    # le recours suivant doit répondre au lieu d'arrêter l'appel.
    suivant = _LLMRepond("20b")
    cascade = LLMEnCascade([_LLMSature(), _LLMSature()], derniers_recours=[_LLMSature(), suivant])
    assert cascade.generer("c", "q") == "20b"
    assert suivant.appels == 1


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


def test_groq_reserve_au_plus_la_limite_de_sortie_du_modele():
    # qwen3.8-27b : 1000 tokens de sortie par minute, une réservation de 1024
    # est refusée d'avance par Groq (2026-11-12).
    assert GroqLLM("fake-key", "qwen/qwen3.8-27b").max_tokens == 1000
    assert GroqLLM("fake-key", "openai/gpt-oss-20b").max_tokens == 1024


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


def test_le_titre_de_la_loi_est_transmis_au_modele():
    llm = _LLMFake(_REPONSE_OK)
    resultat = _Resultat("loi_06_07_1989", "Un logement meublé est un logement décent.", article="25-4")
    resultat.metadata["titre"] = "Loi n° 89-462 du 6 juillet 1989"
    generer_reponse("Question ?", [resultat], llm)
    assert "Texte : Loi n° 89-462 du 6 juillet 1989" in llm.question_recue


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


def test_mots_colles_par_lextraction_du_pdf_sont_toleres():
    # Loi Hoguet, art. 1 (q28 refusée à tort le 2026-10-20) : le texte source
    # porte « suivantsdu », espace perdue à l'extraction.
    article = _Resultat(
        "loi-89-462",
        "8° La conclusion de tout contrat de jouissance d'immeuble à temps partagé régi par les\n"
        "articles L. 121-60 et suivantsdu code de la consommation ;",
        article="15",
    )
    passage = (
        "8° La conclusion de tout contrat de jouissance d'immeuble à temps partagé régi par "
        "les articles L. 121-60 et suivants du code de la consommation ;"
    )
    assert verifier_citations(_citation_longue(passage), [article]).refus is False


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


def test_article_trop_long_reduit_aux_alineas_proches_de_la_question():
    # t34 : l'art. 31 du CGI (51 800 caractères) dépassait seul le budget, la
    # requête dépassait le plafond de Groq (413).
    lignes = ["Article 31"] + [f"Alinéa sans rapport numéro {i} " + "x" * 500 for i in range(100)]
    lignes.insert(60, "b) Les dépenses d'amélioration, à l'exclusion des travaux d'agrandissement ;")
    long_ = _Resultat("code_impot", "\n".join(lignes), article="31")
    llm = _LLMFake(_REPONSE_OK)
    generer_reponse("Les travaux d'agrandissement sont-ils déductibles ?", [long_], llm)
    assert "travaux d'agrandissement ;" in llm.question_recue
    assert "[…]" in llm.question_recue
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


def _verifier_sans_juge(texte: str) -> ReponseGeneree:
    return verifier_fidelite(
        _reponse_avec_citation(texte), [_ARTICLE_15], _NLISelonAffirmation(),
        seuil_haut=0.7, seuil_contradiction=0.7,
    )


def test_affirmation_non_fondee_retiree_le_reste_garde():
    verifiee = _verifier_sans_juge("Le préavis est de trois mois. Le bailleur doit aussi verser une prime.")
    assert verifiee.refus is False
    assert verifiee.reponse == "Le préavis est de trois mois."
    assert verifiee.citations
    assert "prime" in verifiee.motif


def test_premiere_affirmation_non_fondee_fait_refuser():
    # La première phrase porte la réponse directe : sans elle, plus de réponse.
    verifiee = _verifier_sans_juge("Oui, le bailleur doit verser une prime. Le préavis est de trois mois.")
    assert verifiee.refus is True
    assert "prime" in verifiee.motif


def test_toutes_les_affirmations_non_fondees_font_refuser():
    verifiee = _verifier_sans_juge("Le bailleur doit verser une prime.")
    assert verifiee.refus is True


def test_element_de_liste_non_fonde_retire_avec_son_introduction_vide():
    texte = (
        "Le préavis est de trois mois.\n"
        "À noter :\n"
        "- Le bailleur doit verser une prime.\n"
        "Le préavis de trois mois court à réception."
    )
    verifiee = _verifier_sans_juge(texte)
    assert verifiee.refus is False
    assert verifiee.reponse == "Le préavis est de trois mois.\nLe préavis de trois mois court à réception."


def test_element_de_liste_non_fonde_retire_les_autres_gardes():
    texte = (
        "Le préavis est de trois mois.\n"
        "Les règles sont :\n"
        "- un préavis de trois mois ;\n"
        "- une prime au bailleur."
    )
    verifiee = _verifier_sans_juge(texte)
    assert verifiee.reponse == "Le préavis est de trois mois.\nLes règles sont :\n- un préavis de trois mois ;"


def test_montant_absent_retire_avec_sa_phrase():
    texte = "Le préavis est de trois mois. Cumulés, ces délais représentent 100 % du bail."
    verifiee = verifier_fidelite(
        _reponse_avec_citation(texte), [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_contradiction=0.7
    )
    assert verifiee.refus is False
    assert verifiee.reponse == "Le préavis est de trois mois."


def test_montant_absent_dans_la_premiere_phrase_fait_refuser():
    texte = "Le préavis coûte 100 % du loyer. Le préavis est de trois mois."
    verifiee = verifier_fidelite(
        _reponse_avec_citation(texte), [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_contradiction=0.7
    )
    assert verifiee.refus is True
    assert "100 %" in verifiee.motif


def _verifier_guillemets(texte: str) -> ReponseGeneree:
    return verifier_fidelite(
        _reponse_avec_citation(texte), [_ARTICLE_15], _NLIFixe(0.95), seuil_haut=0.7, seuil_contradiction=0.7
    )


def test_fausse_citation_entre_guillemets_fait_refuser():
    # Phrase réécrite qui partage presque toutes ses lettres avec l'article.
    texte = "Selon l'article 15, « lorsque le congé émane du bailleur, le délai de préavis est de trois mois »."
    verifiee = _verifier_guillemets(texte)
    assert verifiee.refus is True
    assert "citation absente" in verifiee.motif


def test_fausse_citation_hors_premiere_phrase_retiree():
    texte = (
        "Le préavis est de trois mois. "
        "La loi précise « lorsque le congé émane du bailleur, le délai de préavis est de trois mois »."
    )
    verifiee = _verifier_guillemets(texte)
    assert verifiee.refus is False
    assert verifiee.reponse == "Le préavis est de trois mois."


def test_citation_exacte_entre_guillemets_acceptee():
    texte = "Le préavis est de trois mois : « lorsque le congé émane du locataire, le délai de préavis est de trois mois »."
    assert _verifier_guillemets(texte).refus is False


def test_guillemets_tolerent_terminaison_crochets_et_terme_court():
    texte = (
        "Le préavis est de trois mois : « lorsque le congé émanent du locataire, le délai [légal] de "
        "préavis est de trois mois ». C'est un « délai de préavis »."
    )
    assert _verifier_guillemets(texte).refus is False


def test_retirer_affirmations_introuvable_renvoie_none():
    assert _retirer_affirmations("Le préavis est de trois mois.", ["Une phrase absente."]) is None


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


def test_ex_et_cf_ne_coupent_pas_la_phrase():
    # q02 : l'affirmation coupée finissait par « (ex. » et le juge la refusait.
    texte = "Le préavis est de trois mois, sauf cas prévus (ex. emploi, santé). Voir cf. article 15."
    assert _decouper_en_affirmations(texte) == [
        "Le préavis est de trois mois, sauf cas prévus (ex. emploi, santé).",
        "Voir cf. article 15.",
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


def test_reference_entre_parentheses_est_retiree():
    # q55 (2026-10-24) : la référence faisait tomber le NLI de 1,00 à 0,00.
    texte = "« Le prix du bail sera diminué. » (article 1724 du code civil). Le préavis est d'un mois (art. 25-8)."
    assert _decouper_en_affirmations(texte) == [
        "« Le prix du bail sera diminué. ».",
        "Le préavis est d'un mois.",
    ]


def test_selon_larticle_en_tete_est_retire():
    texte = "Selon l’article 3 de la loi Hoguet, les activités exigent une carte professionnelle."
    assert _decouper_en_affirmations(texte) == [
        "les activités exigent une carte professionnelle."
    ]


def test_reference_selon_entre_parentheses_est_retiree():
    texte = "Les personnes physiques (selon l'article 3 de la loi Hoguet) justifient d'une garantie."
    assert _decouper_en_affirmations(texte) == [
        "Les personnes physiques justifient d'une garantie."
    ]


def test_parenthese_qui_nest_pas_une_reference_est_gardee():
    texte = "Le préavis est d'un mois (régime du titre Ier bis de la loi du 6 juillet 1989)."
    assert _decouper_en_affirmations(texte) == [texte]


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


def test_reference_avec_identifiant_du_document_nest_pas_une_affirmation():
    texte = "1. Le commandement doit indiquer le délai de deux mois. (loi_06_07_1989, article 24, I‑1)"
    assert _decouper_en_affirmations(texte) == ["Le commandement doit indiquer le délai de deux mois."]


def test_reference_a_une_plage_delements_nest_pas_une_affirmation():
    texte = "   - pour le locataire muté. (Article 15, I, 1°‑5°)"
    assert _decouper_en_affirmations(texte) == ["pour le locataire muté."]


def test_le_juge_recoit_la_reponse_complete_pour_les_renvois():
    texte = "Le préavis est de trois mois.\nCe délai est réduit à un mois en meublé."
    reponse = _reponse_avec_citation(texte)
    juge = _JugeLLM(fondee=True)
    verifier_fidelite(reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge)
    recu = juge.question_recue
    assert texte in recu
    assert recu.index(texte) < recu.index("Affirmations à vérifier :")


def test_acteur_absent_des_articles_cites_part_au_juge():
    # vh05 : « la commune » alors que l'article ne nomme que l'administration.
    reponse = _reponse_avec_citation("La commune dispose d'un délai de préavis de trois mois.")
    juge = _JugeLLM(fondee=False)
    resultat = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert juge.appele
    assert resultat.refus


def test_acteur_present_dans_l_article_ne_derange_pas_le_juge():
    reponse = _reponse_avec_citation("Le locataire donne un préavis de trois mois.")
    juge = _JugeLLM(fondee=False)
    resultat = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert not juge.appele
    assert not resultat.refus


def test_le_juge_recoit_la_question_comme_situation():
    # v089 / v102 : les faits de la situation (« si ma commune a voté ») ne
    # figurent pas dans l'article ; sans la question, le juge rejetait.
    reponse = _reponse_avec_citation("Le préavis est plutôt court.")
    juge = _JugeLLM(fondee=True)
    question = "Je loue en meublé depuis 2025, quel préavis ?"
    verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.5), seuil_haut=0.7, seuil_contradiction=0.7,
        llm=juge, question=question,
    )
    recu = juge.question_recue
    assert question in recu
    assert recu.index(question) < recu.index("Texte source :")


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


_ARTICLE_156 = _Resultat(
    "code_impot",
    "Le déficit foncier s'impute sur le revenu global dans la limite annuelle de 10 700 €. "
    "La limite mentionnée au deuxième alinéa est rehaussée, sans pouvoir excéder 21 400 € par an.",
    article="156",
)


def _reponse_plafond_majore() -> ReponseGeneree:
    return ReponseGeneree(
        refus=False,
        reponse="Le déficit foncier s'impute sur le revenu global jusqu'à 21 400 € par an.",
        citations=[Citation(doc_id="code_impot", article="156", passage="sans pouvoir excéder 21 400 €")],
    )


def test_affirmation_avec_exception_part_au_juge():
    # t05 : « sauf » d'un article greffé sur la règle d'un autre, fondé par le NLI.
    juge = _JugeLLM(fondee=False)
    reponse = _reponse_avec_citation("Le préavis est de trois mois, sauf si le logement est meublé.")
    verifiee = verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.appele is True
    assert verifiee.refus is True


class _JugeExceptions:
    """Juge groupé qui accepte tout, puis seconde lecture des exceptions
    figée : phrase recopiée et verdict donnés."""

    def __init__(self, phrase: str, fondee: bool):
        self._verdict = {"phrase": phrase, "fondee": fondee}
        self.consignes: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        self.consignes.append(consigne)
        n = len(re.findall(r"^\d+\. ", question.split("Affirmations à vérifier :")[-1], re.MULTILINE))
        if consigne == CONSIGNE_JUGE_EXCEPTION:
            return json.dumps({"verdicts": [self._verdict] * n})
        return json.dumps({"fondees": [True] * n})


def _verifier_exception(juge: _JugeExceptions) -> ReponseGeneree:
    reponse = _reponse_avec_citation("Le préavis est de trois mois, sauf si le logement est meublé.")
    return verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )


def test_exception_rattachee_a_une_autre_regle_est_rejetee_a_la_seconde_lecture():
    # t05 : le juge groupé acceptait l'exception de l'art. 22-1 greffée sur l'art. 22-2.
    juge = _JugeExceptions("le délai de préavis est de trois mois", fondee=False)
    assert _verifier_exception(juge).refus is True
    assert juge.consignes == [CONSIGNE_JUGE_FIDELITE, CONSIGNE_JUGE_EXCEPTION]


def test_seconde_lecture_sans_phrase_retrouvee_garde_le_verdict_du_juge():
    juge = _JugeExceptions("phrase absente des articles cités", fondee=False)
    assert _verifier_exception(juge).refus is False


def test_seconde_lecture_qui_confirme_garde_la_reponse():
    juge = _JugeExceptions("le délai de préavis est de trois mois", fondee=True)
    assert _verifier_exception(juge).refus is False


def test_pas_de_seconde_lecture_sans_exception():
    juge = _JugeExceptions("", fondee=False)
    reponse = _reponse_avec_citation("Le préavis est de trois mois.")
    verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.1), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.consignes == [CONSIGNE_JUGE_FIDELITE]


class _JugeReponseDirecte:
    """Juge groupé qui accepte tout, puis seconde lecture de la réponse
    directe figée : phrase recopiée et verdict donnés."""

    def __init__(self, phrase: str, fondee: bool):
        self._verdict = {"phrase": phrase, "fondee": fondee}
        self.consignes: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        self.consignes.append(consigne)
        n = len(re.findall(r"^\d+\. ", question.split("Affirmations à vérifier :")[-1], re.MULTILINE))
        if consigne == CONSIGNE_JUGE_REPONSE_DIRECTE:
            return json.dumps({"verdicts": [self._verdict] * n})
        return json.dumps({"fondees": [True] * n})


_REPONSE_NEGATIVE_PUIS_CONDITION = (
    "Le locataire n'a pas de préavis à respecter. "
    "Lorsque le congé émane du locataire, le délai de préavis est de trois mois."
)


def test_reponse_directe_negative_contredite_par_ses_conditions_est_refusee():
    # n05 : « ne reste pas tenu » en tête, puis la solidarité qui dure six mois.
    juge = _JugeReponseDirecte("le délai de préavis est de trois mois", fondee=False)
    verifiee = verifier_fidelite(
        _reponse_avec_citation(_REPONSE_NEGATIVE_PUIS_CONDITION), [_ARTICLE_15], _NLIFixe(0.1),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert verifiee.refus is True
    assert juge.consignes == [CONSIGNE_JUGE_FIDELITE, CONSIGNE_JUGE_REPONSE_DIRECTE]


def test_reponse_directe_confirmee_a_la_relecture_est_gardee():
    juge = _JugeReponseDirecte("le délai de préavis est de trois mois", fondee=True)
    verifiee = verifier_fidelite(
        _reponse_avec_citation(_REPONSE_NEGATIVE_PUIS_CONDITION), [_ARTICLE_15], _NLIFixe(0.1),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert verifiee.refus is False


def test_pas_de_relecture_de_la_reponse_directe_positive():
    juge = _JugeReponseDirecte("", fondee=False)
    reponse = _reponse_avec_citation(
        "Le locataire doit un préavis. Lorsque le congé émane du locataire, le délai de préavis est de trois mois."
    )
    verifier_fidelite(
        reponse, [_ARTICLE_15], _NLIFixe(0.1), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.consignes == [CONSIGNE_JUGE_FIDELITE]


def test_pas_de_relecture_de_la_reponse_directe_fondee_par_le_nli():
    juge = _JugeReponseDirecte("", fondee=False)
    verifier_fidelite(
        _reponse_avec_citation(_REPONSE_NEGATIVE_PUIS_CONDITION), [_ARTICLE_15], _NLIFixe(0.99),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.consignes == []


def test_affirmation_fondee_sur_une_derogation_part_au_juge():
    # Vraie mot à mot, l'affirmation tirée de « est rehaussée » peut taire la
    # règle générale : le NLI seul ne suffit pas (q58).
    juge = _JugeLLM(fondee=False)
    verifiee = verifier_fidelite(
        _reponse_plafond_majore(), [_ARTICLE_156], _NLIFixe(0.99),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )
    assert juge.appele is True
    assert verifiee.refus is True


def _verdict_incomplet(regle: str) -> str:
    return json.dumps({"complete": False, "regle_generale": regle, "manque": "Règle générale : 10 700 €."})


def test_completude_renvoie_l_element_omis_et_sa_preuve():
    llm = _LLMFake(_verdict_incomplet("dans la limite annuelle de 10 700 €"))
    manque = verifier_completude("Jusqu'à quel montant ?", _reponse_plafond_majore(), [_ARTICLE_156], llm)
    assert manque == "Règle générale : 10 700 €. (« dans la limite annuelle de 10 700 € »)"
    # Le relecteur reçoit la question, la réponse et l'article cité.
    assert "Jusqu'à quel montant ?" in llm.question_recue
    assert "21 400 €" in llm.question_recue
    assert "10 700 €" in llm.question_recue


def test_completude_reponse_complete():
    llm = _LLMFake(json.dumps({"complete": True, "manque": ""}))
    assert verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156], llm) is None


@pytest.mark.parametrize("brut", ["pas du json", "[]", json.dumps({"complete": False, "manque": ""})])
def test_completude_verdict_inexploitable_vaut_complete(brut):
    # Un échec technique ne transforme pas une réponse déjà vérifiée en refus.
    assert verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156], _LLMFake(brut)) is None


def test_completude_refus_sans_appel():
    llm = _LLMFake("ne doit pas servir")
    assert verifier_completude("Q ?", ReponseGeneree(refus=True, motif="x"), [_ARTICLE_156], llm) is None
    assert llm.consigne_recue is None


def test_completude_phrase_introuvable_dans_les_articles_ignoree():
    # Omission sans appui dans le texte récupéré (article inventé) : ignorée.
    llm = _LLMFake(_verdict_incomplet("Le locataire peut résilier le bail sans préavis."))
    assert verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156], llm) is None


def test_completude_regle_cherchee_dans_un_article_non_cite():
    # La règle générale peut être dans un article récupéré que la réponse ne
    # cite pas (q02 : seul l'article des meublés était cité).
    article_15 = _Resultat("loi-89-462", "Le délai de préavis est de trois mois.", article="15")
    llm = _LLMFake(_verdict_incomplet("Le délai de préavis est de trois mois."))
    manque = verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156, article_15], llm)
    assert manque is not None
    assert "Article 15" in llm.question_recue


def test_completude_signale_un_passage_cite_de_derogation():
    # Le juge laissait passer le plafond majoré seul une fois sur trois (q58) :
    # un passage cité qui dit « rehaussée » lui est signalé.
    reponse = _reponse_plafond_majore()
    reponse.citations[0].passage = "est rehaussée, sans pouvoir excéder 21 400 € par an"
    llm = _LLMFake(json.dumps({"complete": True}))
    verifier_completude("Q ?", reponse, [_ARTICLE_156], llm)
    assert "énoncent une dérogation" in llm.question_recue
    assert "« est rehaussée, sans pouvoir excéder 21 400 € par an »" in llm.question_recue


def test_completude_sans_derogation_pas_de_signal():
    llm = _LLMFake(json.dumps({"complete": True}))
    verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156], llm)
    assert "dérogation" not in llm.question_recue


_SECTION_DEUX = "Section 2 : Dispositions particulières aux syndicats dont le nombre de voix est réparti entre deux copropriétaires"


def _reponse_deux_coproprietaires() -> tuple[ReponseGeneree, _Resultat]:
    article = _Resultat(
        "loi-65-557", "Les décisions peuvent être prises par le copropriétaire détenant plus de la moitié des voix.",
        article="41-16", section=f"Chapitre IV ter : Dispositions particulières à certaines copropriétés > {_SECTION_DEUX}",
    )
    reponse = ReponseGeneree(
        refus=False, reponse="Oui, il peut décider seul.",
        citations=[Citation(doc_id="loi-65-557", article="41-16",
                            passage="peuvent être prises par le copropriétaire détenant plus de la moitié des voix")],
    )
    return reponse, article


def test_completude_signale_l_intitule_d_un_regime_particulier_cite():
    # t15 : l'art. 41-16 (deux copropriétaires) présenté comme règle commune.
    reponse, article = _reponse_deux_coproprietaires()
    llm = _LLMFake(json.dumps({"complete": True}))
    verifier_completude("Q ?", reponse, [article], llm)
    assert "relèvent d'un régime particulier" in llm.question_recue
    assert f"« {_SECTION_DEUX} »" in llm.question_recue


def test_completude_omission_sans_preuve_sur_regime_particulier():
    # t15 : le juge voit l'omission mais recopie une règle absente des
    # articles récupérés ; l'intitulé du régime en tient lieu.
    reponse, article = _reponse_deux_coproprietaires()
    llm = _LLMFake(_verdict_incomplet("Règle inventée absente des articles."))
    manque = verifier_completude("Q ?", reponse, [article], llm)
    assert manque == f"L'article 41-16 relève de : « {_SECTION_DEUX} »."


def test_completude_intitule_accepte_comme_preuve():
    # La règle générale (art. 22) n'était pas récupérée : l'intitulé qui borne
    # le régime suffit à prouver l'omission.
    reponse, article = _reponse_deux_coproprietaires()
    llm = _LLMFake(_verdict_incomplet(_SECTION_DEUX))
    assert verifier_completude("Q ?", reponse, [article], llm) is not None


def test_completude_a_verifier_passage_de_derogation():
    reponse = _reponse_plafond_majore()
    reponse.citations[0].passage = "est rehaussée, sans pouvoir excéder 21 400 € par an"
    assert completude_a_verifier(reponse, [_ARTICLE_156]) is True


def _reponse_parking() -> tuple[ReponseGeneree, _Resultat]:
    # v065 : « Dans ce cas » renvoie à la clause de priorité du règlement.
    article = _Resultat(
        "loi-65-557",
        "Le règlement de copropriété peut prévoir une clause attribuant un droit de priorité. "
        "Dans ce cas, le vendeur doit faire connaître au syndic son intention de vendre.",
        article="8-1",
    )
    reponse = ReponseGeneree(
        refus=False, reponse="Oui, le vendeur doit prévenir le syndic.",
        citations=[Citation(doc_id="loi-65-557", article="8-1",
                            passage="Dans ce cas, le vendeur doit faire connaître au syndic son intention de vendre.")],
    )
    return reponse, article


def test_completude_a_verifier_passage_qui_renvoie_au_cas_precedent():
    reponse, article = _reponse_parking()
    assert completude_a_verifier(reponse, [article]) is True


def test_completude_signale_un_passage_conditionnel():
    reponse, article = _reponse_parking()
    llm = _LLMFake(json.dumps({"complete": True}))
    verifier_completude("Q ?", reponse, [article], llm)
    assert "cas posé juste avant" in llm.question_recue


def test_completude_a_verifier_article_de_regime_particulier():
    # q02 : seul l'article des meublés est cité, la règle générale est ailleurs.
    meuble = _Resultat("loi-89-462", "Le délai de préavis est d'un mois.", article="25-8",
                       section="Titre Ier bis : Des rapports entre bailleurs et locataires dans les logements meublés")
    reponse = ReponseGeneree(
        refus=False, reponse="Le préavis est d'un mois.",
        citations=[Citation(doc_id="loi-89-462", article="25-8", passage="Le délai de préavis est d'un mois.")],
    )
    assert completude_a_verifier(reponse, [meuble]) is True


def test_completude_a_verifier_article_cite_trop_long():
    # t34 : l'art. 31 du CGI mêle règle générale et régimes optionnels.
    long_ = _Resultat("code_impot", "z" * (LONGUEUR_MAX_CONTEXTE_CARACTERES + 1), article="31")
    reponse = ReponseGeneree(
        refus=False, reponse="Oui.",
        citations=[Citation(doc_id="code_impot", article="31", passage="zzz")],
    )
    assert completude_a_verifier(reponse, [long_]) is True


_ARTICLE_1415 = _Resultat(
    "code_impot", "La taxe foncière est établie pour l'année entière d'après les faits existants au 1er janvier.",
    article="1415",
)
_ARTICLE_1403 = _Resultat(
    "code_impot", "Tant que la mutation cadastrale n'a pas été faite, l'ancien propriétaire continue à être imposé.",
    article="1403",
)


def _reponse_mutation() -> ReponseGeneree:
    return ReponseGeneree(
        refus=False, reponse="Le vendeur paie jusqu'à la mutation cadastrale.",
        citations=[Citation(doc_id="code_impot", article="1403", passage="l'ancien propriétaire continue à être imposé")],
    )


def test_completude_a_verifier_premier_article_non_cite():
    # t37 : l'art. 1415, classé premier, tu au profit de l'art. 1403.
    assert completude_a_verifier(_reponse_mutation(), [_ARTICLE_1415, _ARTICLE_1403]) is True
    assert completude_a_verifier(_reponse_mutation(), [_ARTICLE_1403, _ARTICLE_1415]) is False


def test_completude_signale_le_premier_article_non_cite():
    llm = _LLMFake(json.dumps({"complete": True}))
    verifier_completude("Q ?", _reponse_mutation(), [_ARTICLE_1415, _ARTICLE_1403], llm)
    assert "l'article 1415, classé premier" in llm.question_recue


def test_completude_a_verifier_reponse_ordinaire_ou_refus():
    assert completude_a_verifier(_reponse_plafond_majore(), [_ARTICLE_156]) is False
    assert completude_a_verifier(ReponseGeneree(refus=True, motif="x"), [_ARTICLE_156]) is False


def test_decoupage_garde_le_guillemet_fermant_avec_sa_phrase():
    # q55 : « … louée. » (article 1724 du code civil). laissait « » (article …) » seul.
    texte = "Principe : « Le preneur doit les souffrir. » (article 1724 du code civil). Suite du texte."
    assert _decouper_en_affirmations(texte) == [
        "Principe : « Le preneur doit les souffrir. ».",
        "Suite du texte.",
    ]


def test_decoupage_ignore_intertitres_en_gras_et_questions():
    # q54 : intertitres gras numérotés et question rhétorique, sans contenu à vérifier.
    texte = (
        "**Non.** L'agent ne peut pas encaisser l'acompte.\n\n"
        "**2. Interdiction de recevoir des fonds**\n"
        "Les personnes habilitées ne peuvent pas recevoir de fonds.\n\n"
        "**3. Qui peut encaisser l'acompte ?**\n"
        "Le titulaire de la carte peut être autorisé à recevoir des fonds."
    )
    assert _decouper_en_affirmations(texte) == [
        "L'agent ne peut pas encaisser l'acompte.",
        "Les personnes habilitées ne peuvent pas recevoir de fonds.",
        "Le titulaire de la carte peut être autorisé à recevoir des fonds.",
    ]


def test_decoupage_coupe_apres_le_guillemet_fermant_avant_une_nouvelle_phrase():
    texte = "Selon l'article 1721, « Il est dû garantie. » De plus, le bailleur est tenu."
    assert _decouper_en_affirmations(texte) == [
        "« Il est dû garantie. »",
        "De plus, le bailleur est tenu.",
    ]


def test_montant_deforme_bascule_en_refus():
    # q48 (2026-10-23) : « 1 300 € » pour « 1 300 000 € », validé par le NLI.
    article = _Resultat(
        "code_impot",
        "Supérieure à 800 000 € et inférieure ou égale à 1 300\n000 €\n0,50",
        article="977",
    )
    reponse = ReponseGeneree(
        refus=False,
        reponse="Le taux est de 0,5 % entre 800 001 € et 1 300 €.",
        citations=[Citation(doc_id="code_impot", article="977", passage="Supérieure à 800 000 €")],
    )
    verdict = verifier_fidelite(reponse, [article], _NLIFixe(0.99), 0.5, 0.7)
    assert verdict.refus is True
    assert "1 300 €" in verdict.motif


def test_montants_presents_dans_l_article_acceptes():
    article = _Resultat(
        "code_impot", "Supérieure à 800 000 € et inférieure ou égale à 1 300\n000 €\n0,50", article="977"
    )
    reponse = ReponseGeneree(
        refus=False,
        reponse="Le taux est de 0,50 % entre 800 001 € et 1 300 000 €.",
        citations=[Citation(doc_id="code_impot", article="977", passage="Supérieure à 800 000 €")],
    )
    assert verifier_fidelite(reponse, [article], _NLIFixe(0.99), 0.5, 0.7).refus is False


def test_montant_repris_de_la_question_accepte():
    # v008 (2026-11-02) : « 15 % » vient de la question, pas de l'article.
    article = _Resultat(
        "loi_1989", "Toutefois, si la hausse est supérieure à 10 %, elle s'applique par tiers annuel.",
        article="25-9",
    )
    reponse = ReponseGeneree(
        refus=False,
        reponse="Non, la hausse de 15 % s'applique par tiers annuel.",
        citations=[Citation(doc_id="loi_1989", article="25-9", passage="elle s'applique par tiers annuel")],
    )
    question = "Mon loyer augmente de 15 % : dois-je tout payer immédiatement ?"
    assert verifier_fidelite(reponse, [article], _NLIFixe(0.99), 0.5, 0.7).refus is True
    assert verifier_fidelite(
        reponse, [article], _NLIFixe(0.99), 0.5, 0.7, question=question
    ).refus is False


_ARTICLE_41_12 = _Resultat(
    "loi_1965",
    "Par dérogation aux dispositions de l'article 17, les décisions, à l'exclusion de celles relatives\n"
    "au vote du budget prévisionnel et à l'approbation des comptes, peuvent être prises à\n"
    "l'unanimité des voix des copropriétaires à l'occasion d'une consultation écrite.",
    article="41-12",
)


def _reponse_citant(passage: str) -> ReponseGeneree:
    return ReponseGeneree(
        refus=False,
        reponse="Le budget est voté en assemblée.",
        citations=[Citation(doc_id="loi_1965", article="41-12", passage=passage)],
    )


def test_citation_amputee_dune_incise_est_remplacee_par_la_phrase_entiere():
    # q20 (2026-10-22) : « et à l'approbation des comptes » retiré au milieu.
    passage = (
        "Par dérogation aux dispositions de l'article 17, les décisions, **à l'exclusion de celles "
        "relatives au vote du budget prévisionnel**, peuvent être prises à l'unanimité des voix des "
        "copropriétaires à l'occasion d'une consultation écrite."
    )
    reparee = reparer_citations(_reponse_citant(passage), [_ARTICLE_41_12])
    assert "et à l'approbation des comptes" in reparee.citations[0].passage
    assert not verifier_citations(reparee, [_ARTICLE_41_12]).refus


def test_citation_qui_ajoute_un_mot_nest_pas_reparee():
    passage = (
        "Par dérogation aux dispositions de l'article 17, les décisions, y compris celles relatives "
        "au vote du budget prévisionnel, peuvent être prises à l'unanimité des voix."
    )
    reparee = reparer_citations(_reponse_citant(passage), [_ARTICLE_41_12])
    assert reparee.citations[0].passage == passage


def test_fragment_trop_court_nest_pas_repare():
    passage = "les décisions peuvent être prises à l'unanimité"
    reparee = reparer_citations(_reponse_citant(passage), [_ARTICLE_41_12])
    assert reparee.citations[0].passage == passage


def test_ligature_oe_citee_retrouvee_dans_larticle_ecrit_oe():
    # q25 (2026-10-06) : « mise en œuvre » cité, « mise en oeuvre » dans l'article.
    article = _Resultat("loi_1965", "L'opposition vaut mise en oeuvre de l'hypothèque légale.", article="20")
    reponse = ReponseGeneree(
        refus=False,
        reponse="L'opposition vaut hypothèque.",
        citations=[Citation(doc_id="loi_1965", article="20", passage="L'opposition vaut mise en œuvre de l'hypothèque légale.")],
    )
    assert not verifier_citations(reponse, [article]).refus


def test_introduction_et_element_denumeration_recolles_sont_retrouves():
    # q17 : l'introduction et le 2° sont exacts, mais séparés dans l'article par le 1°.
    article = _Resultat(
        "loi_1989",
        "Le commandement de payer contient, à peine de nullité :\n"
        "1° La mention que le locataire dispose d'un délai de six semaines ;\n"
        "2° Le montant mensuel du loyer et des charges ;",
        article="24",
    )
    passage = "Le commandement de payer contient, à peine de nullité : 2° Le montant mensuel du loyer et des charges ;"
    reponse = ReponseGeneree(
        refus=False,
        reponse="Le commandement indique le loyer.",
        citations=[Citation(doc_id="loi_1989", article="24", passage=passage)],
    )
    assert not verifier_citations(reponse, [article]).refus


def test_element_denumeration_altere_reste_refuse():
    article = _Resultat(
        "loi_1989",
        "Le commandement de payer contient, à peine de nullité :\n2° Le montant mensuel du loyer et des charges ;",
        article="24",
    )
    passage = "Le commandement de payer contient, à peine de nullité : 2° Le montant annuel des dépôts versés ;"
    reponse = ReponseGeneree(
        refus=False,
        reponse="Le commandement indique les dépôts.",
        citations=[Citation(doc_id="loi_1989", article="24", passage=passage)],
    )
    assert verifier_citations(reponse, [article]).refus


def test_exemple_chiffre_et_sa_liste_sont_retires():
    # q48 (2026-10-22) : exemple de calcul inventé, refusé pour « 1 500 000 € ».
    texte = (
        "Le taux est de 0,50 % entre 800 000 € et 1 300 000 €.\n\n"
        "**Exemple de calcul** (pour un patrimoine de **1 500 000 €**) :\n"
        "- **0 %** sur les **800 000 €** premiers.\n"
        "- **0,50 %** sur la tranche de **500 000 €**.\n\n"
        "Le barème s'applique à la valeur nette."
    )
    assert retirer_exemples(texte) == (
        "Le taux est de 0,50 % entre 800 000 € et 1 300 000 €.\n\n"
        "Le barème s'applique à la valeur nette."
    )


def test_sous_element_exemple_ne_retire_pas_lelement_suivant():
    # q13 : « 2. » après un sous-élément « - Exemple : » est une autre règle.
    texte = (
        "1. **Reprise** : le bailleur justifie sa décision.\n"
        "   - Exemple : reprise pour une mutation.\n\n"
        "2. **Vente par lots** : le bail peut être reconduit."
    )
    assert retirer_exemples(texte) == (
        "1. **Reprise** : le bailleur justifie sa décision.\n\n"
        "2. **Vente par lots** : le bail peut être reconduit."
    )


def test_par_exemple_au_fil_de_la_phrase_est_garde():
    texte = "Le préavis est réduit, par exemple en cas de mutation professionnelle."
    assert retirer_exemples(texte) == texte


def test_reponse_faite_seulement_dun_exemple_reste_entiere():
    reponse = ReponseGeneree(refus=False, reponse="**Exemple** : un loyer de 1 000 €.", citations=[])
    assert sans_exemples(reponse).reponse == reponse.reponse


def test_introduction_avec_sujet_reste_rattachee():
    texte = "Le locataire doit :\n- payer le loyer."
    assert _decouper_en_affirmations(texte) == ["Le locataire doit notamment : payer le loyer."]


def test_notamment_nest_pas_double():
    texte = "Elle permet notamment :\n- de résilier le bail."
    assert _decouper_en_affirmations(texte) == ["Elle permet notamment : de résilier le bail."]


_ARTICLE_2_TERDECIES_G = _Resultat(
    "code_impot3",
    "Article 2 terdecies G\nPour l'application du o du 1° du I de l'article 31 du code général des "
    "impôts :\n1° Le délai de préavis est de trois mois.",
    article="2 terdecies G",
)


def test_champ_application_lu_en_tete_d_article():
    assert champ_application(_ARTICLE_2_TERDECIES_G.content) == (
        "Pour l'application du o du 1° du I de l'article 31 du code général des impôts"
    )
    assert champ_application(_ARTICLE_15.content) is None


def test_champ_application_montre_au_modele():
    assert "Champ d'application : Pour l'application du o" in _bloc_articles([_ARTICLE_2_TERDECIES_G])


def test_article_a_champ_d_application_part_au_juge():
    # vh13 : plafonds d'une déduction fiscale présentés comme ceux du logement social.
    reponse = _reponse_avec_citation("Le délai de préavis est de trois mois.", article="2 terdecies G")
    reponse.citations[0].doc_id = "code_impot3"
    juge = _JugeLLM(fondee=False)
    resultat = verifier_fidelite(
        reponse, [_ARTICLE_2_TERDECIES_G], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge
    )
    assert juge.appele
    assert resultat.refus
    assert completude_a_verifier(reponse, [_ARTICLE_2_TERDECIES_G])


def test_rappel_du_champ_quand_la_reponse_ne_le_nomme_pas():
    reponse = _reponse_avec_citation("Le délai de préavis est de trois mois.", article="2 terdecies G")
    reponse.citations[0].doc_id = "code_impot3"
    complete = rappeler_champ(reponse, [_ARTICLE_2_TERDECIES_G])
    assert "Précision : L'article 2 terdecies G vaut seulement « Pour l'application du o" in complete.reponse
    nomme = reponse.model_copy(update={"reponse": "Pour la déduction de l'article 31, le délai est de trois mois."})
    assert rappeler_champ(nomme, [_ARTICLE_2_TERDECIES_G]).reponse == nomme.reponse


class _JugeOuiNon:
    """Juge qui ne répond qu'à la relecture du « Oui » / « Non » de tête, avec
    un verdict figé, et retient les consignes reçues."""

    def __init__(self, coherent: bool):
        self._coherent = coherent
        self.consignes: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        self.consignes.append(consigne)
        return json.dumps({"coherent": self._coherent})


_QUESTION_DISPENSE = "Le locataire doit-il respecter un préavis ?"
_OUI_SUIVI_DUNE_DISPENSE = "Oui, le locataire peut partir sans préavis de trois mois."


def _verifier_oui_non(texte: str, juge, question: str = _QUESTION_DISPENSE) -> ReponseGeneree:
    # NLI à 0,99 : les affirmations sont fondées sans juge, seule la relecture
    # du « Oui » / « Non » peut l'appeler.
    return verifier_fidelite(
        _reponse_avec_citation(texte), [_ARTICLE_15], _NLIFixe(0.99),
        seuil_haut=0.7, seuil_contradiction=0.7, llm=juge, question=question,
    )


def test_oui_contraire_a_la_question_est_retire():
    # v178 : « doit-il suivre la procédure ? » → « Oui, […] sans suivre la procédure ».
    juge = _JugeOuiNon(coherent=False)
    verifiee = _verifier_oui_non(_OUI_SUIVI_DUNE_DISPENSE, juge)
    assert verifiee.refus is False
    assert verifiee.reponse == "Le locataire peut partir sans préavis de trois mois."
    assert juge.consignes == [CONSIGNE_OUI_NON]


def test_oui_juge_coherent_est_garde():
    verifiee = _verifier_oui_non(_OUI_SUIVI_DUNE_DISPENSE, _JugeOuiNon(coherent=True))
    assert verifiee.reponse == _OUI_SUIVI_DUNE_DISPENSE


def test_reponse_illisible_du_juge_garde_le_oui():
    class _JugeCasse:
        def generer(self, consigne, question):
            return "pas du JSON"

    verifiee = _verifier_oui_non(_OUI_SUIVI_DUNE_DISPENSE, _JugeCasse())
    assert verifiee.reponse == _OUI_SUIVI_DUNE_DISPENSE


@pytest.mark.parametrize(
    "reponse, question",
    [
        # « Oui » suivi d'une phrase affirmative : rien à relire.
        ("Oui, le préavis est de trois mois.", _QUESTION_DISPENSE),
        # « ne … que » restreint sans nier (q26).
        ("Oui, le préavis ne court qu'à compter de la réception.", _QUESTION_DISPENSE),
        # « Non » suivi d'une négation : cohérent.
        ("Non, le préavis ne peut pas être réduit, car la loi prévoit que trois mois.", _QUESTION_DISPENSE),
        # Question négative : le sens attendu s'inverse, pas de relecture.
        (_OUI_SUIVI_DUNE_DISPENSE, "Le locataire n'est-il pas tenu à un préavis ?"),
        # Sans « Oui » / « Non » en tête.
        ("Le locataire peut partir sans préavis de trois mois.", _QUESTION_DISPENSE),
    ],
)
def test_pas_de_relecture_du_oui_non_sans_contradiction_apparente(reponse, question):
    juge = _JugeOuiNon(coherent=False)
    assert oui_non_a_relire(reponse, question) is False
    _verifier_oui_non(reponse, juge, question)
    assert juge.consignes == []


def test_pas_de_relecture_du_oui_non_sans_question():
    juge = _JugeOuiNon(coherent=False)
    _verifier_oui_non(_OUI_SUIVI_DUNE_DISPENSE, juge, question="")
    assert juge.consignes == []


# --- Sens des relations entre personnes (v150) -------------------------------

_ARTICLE_8_2 = _Resultat(
    "loi-89-462",
    "Le fait pour le locataire auteur des violences de ne pas acquitter son loyer est un "
    "motif légitime et sérieux. Le bailleur peut donner congé à son locataire.",
    article="8-2",
)


class _JugeRoles:
    """Extraction des rôles figée : phrase recopiée, qui agit, envers qui."""

    def __init__(self, phrase: str, agit: str, envers: str):
        self._action = {"phrase": phrase, "agit": agit, "envers": envers}
        self.consignes: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        self.consignes.append(consigne)
        if consigne == CONSIGNE_ROLES:
            n = len(re.findall(r"^\d+\. ", question.split("Actions :")[-1], re.MULTILINE))
            return json.dumps({"actions": [self._action] * n})
        n = len(re.findall(r"^\d+\. ", question.split("Affirmations à vérifier :")[-1], re.MULTILINE))
        return json.dumps({"fondees": [True] * n})


def _verifier_roles(texte: str, juge: _JugeRoles) -> ReponseGeneree:
    reponse = ReponseGeneree(
        refus=False,
        reponse=texte,
        citations=[Citation(doc_id="loi-89-462", article="8-2", passage="est un motif légitime et sérieux")],
    )
    return verifier_fidelite(
        reponse, [_ARTICLE_8_2], _NLIFixe(0.99), seuil_haut=0.7, seuil_contradiction=0.7, llm=juge,
    )


def test_relation_inversee_est_rejetee():
    # v150 : c'est le bailleur qui donne congé, pas le locataire « au bailleur ».
    juge = _JugeRoles("Le bailleur peut donner congé à son locataire.", "bailleur", "locataire")
    verifiee = _verifier_roles("C'est un motif légitime et sérieux pour donner congé au bailleur.", juge)
    assert verifiee.refus is True
    assert juge.consignes == [CONSIGNE_ROLES]


def test_relation_inversee_retiree_le_reste_garde():
    juge = _JugeRoles("Le bailleur peut donner congé à son locataire.", "bailleur", "locataire")
    verifiee = _verifier_roles(
        "Le défaut de paiement est un motif légitime et sérieux. Il permet de donner congé au bailleur.",
        juge,
    )
    assert verifiee.refus is False
    assert "au bailleur" not in verifiee.reponse


def test_relation_dans_le_bon_sens_est_gardee():
    juge = _JugeRoles("Le bailleur peut donner congé à son locataire.", "locataire", "bailleur")
    assert _verifier_roles("Le locataire peut donner congé au bailleur.", juge).refus is False


def test_relation_sans_phrase_retrouvee_est_gardee():
    juge = _JugeRoles("Phrase absente des articles cités.", "bailleur", "locataire")
    assert _verifier_roles("C'est un motif pour donner congé au bailleur.", juge).refus is False


def test_pas_d_extraction_sans_verbe_d_action():
    # n08 : « impose au bailleur » ne fait agir personne envers personne.
    juge = _JugeRoles("", "", "")
    _verifier_roles("Ce motif légitime et sérieux impose au bailleur une démarche.", juge)
    assert CONSIGNE_ROLES not in juge.consignes


# --- Situation de la question et cas de l'article cité (v108) -----------------

_ART_555 = _Resultat(
    "code_civil",
    "Lorsque les constructions ont été faites par un tiers avec des matériaux appartenant à ce dernier, "
    "le propriétaire du fonds a le droit d'obliger le tiers à les enlever.",
    article="555",
)
_ART_553 = _Resultat("code_civil", "Toutes constructions sont présumées faites par le propriétaire.", article="553")
_PHRASE_555 = "Lorsque les constructions ont été faites par un tiers avec des matériaux appartenant à ce dernier"


def _reponse_555() -> ReponseGeneree:
    return ReponseGeneree(
        refus=False,
        reponse="Le propriétaire du fonds peut obliger le tiers à enlever les constructions.",
        citations=[Citation(doc_id="code_civil", article="555", passage="obliger le tiers à les enlever")],
    )


class _LLMRolesFige:
    def __init__(self, personnes_article: list[str], phrase: str = _PHRASE_555):
        self._article = {"phrase": phrase, "roles": [{"role": "x", "personne": p} for p in personnes_article]}

    def generer(self, consigne: str, question: str) -> str:
        if consigne == CONSIGNE_ROLES_QUESTION:
            return json.dumps({"roles": [
                {"role": "propriétaire du terrain", "personne": "le voisin"},
                {"role": "auteur de la construction", "personne": "Le voisin"},
            ]})
        assert consigne == CONSIGNE_ROLES_ARTICLE
        return json.dumps(self._article)


_Q108 = "Mon voisin a construit sur son terrain avec mes matériaux, puis-je les reprendre ?"


def test_roles_ecart_signale_au_redacteur():
    ecart = verifier_roles(_Q108, _reponse_555(), [_ART_555], _LLMRolesFige(["le propriétaire du fonds", "un tiers"]))
    assert ecart is not None
    assert "personnes différentes" in ecart and "article 555" in ecart


def test_roles_meme_schema_pas_d_ecart():
    llm = _LLMRolesFige(["le propriétaire du fonds", "Le propriétaire du fonds"])
    assert verifier_roles(_Q108, _reponse_555(), [_ART_555], llm) is None


def test_roles_sans_phrase_retrouvee_pas_d_ecart():
    llm = _LLMRolesFige(["le propriétaire du fonds", "un tiers"], phrase="Phrase inventée par le modèle.")
    assert verifier_roles(_Q108, _reponse_555(), [_ART_555], llm) is None


def test_roles_passages_cites_montres_au_lecteur_de_l_article():
    # x024 : le lecteur doit décrire le cas du passage cité, pas une autre
    # phrase de l'article.
    messages: list[str] = []

    class _Espion(_LLMRolesFige):
        def generer(self, consigne, question):
            messages.append(question)
            return super().generer(consigne, question)

    verifier_roles(_Q108, _reponse_555(), [_ART_555], _Espion(["le propriétaire du fonds", "un tiers"]))
    assert "Passages cités :\n« obliger le tiers à les enlever »" in messages[-1]


def test_roles_designation_collective_ignoree():
    # v023 : « les parties » ne dit pas si deux rôles vont à la même personne.
    llm = _LLMRolesFige(["les parties", "un tiers"])
    assert verifier_roles(_Q108, _reponse_555(), [_ART_555], llm) is None


def test_roles_a_verifier_situation_personnelle_et_article_cite_hors_tete():
    assert roles_a_verifier(_Q108, _reponse_555(), [_ART_553, _ART_555]) is True
    assert roles_a_verifier(_Q108, _reponse_555(), [_ART_555, _ART_553]) is False
    question = "Un tiers construit sur le terrain d'autrui : que prévoit la loi ?"
    assert roles_a_verifier(question, _reponse_555(), [_ART_553, _ART_555]) is False


def test_liste_json_reunit_une_liste_d_objets_par_role():
    # v108 (2026-11-10) : un objet {"phrase", "roles"} par rôle au lieu d'un seul.
    from app.rag.generator import _liste_json

    brut = json.dumps([
        {"phrase": "p", "roles": [{"role": "a", "personne": "x"}]},
        {"phrase": "p", "roles": [{"role": "b", "personne": "y"}]},
    ])
    objet, roles = _liste_json(brut, "roles")
    assert objet["phrase"] == "p"
    assert [r["role"] for r in roles] == ["a", "b"]


# --- complétude : exonération appliquée à la situation (v090) ----------------

_SECTION_TF = "Section II : Taxes foncières > I : Taxe foncière sur les propriétés bâties"
_ART_1381 = _Resultat(
    "code_impot",
    "Article 1381\nSont également soumis à la taxe foncière sur les propriétés bâties : "
    "1° Les installations destinées à abriter des personnes ou des biens ou à stocker des produits ;",
    article="1381",
    section=f"{_SECTION_TF} > A : Propriétés imposables",
)
_ART_1382 = _Resultat(
    "code_impot",
    "Article 1382\nSont exonérés de la taxe foncière sur les propriétés bâties : "
    "6° a. Les bâtiments qui servent aux exploitations rurales tels que granges, écuries ;",
    article="1382",
    section=f"{_SECTION_TF} > B : Exonérations permanentes",
)


def _reponse_exoneree() -> ReponseGeneree:
    return ReponseGeneree(
        refus=False,
        reponse="Les écuries et les granges sont exonérées de la taxe foncière.",
        citations=[
            Citation(
                doc_id="code_impot",
                article="1382",
                passage="Les bâtiments qui servent aux exploitations rurales tels que granges, écuries",
            )
        ],
    )


class _LLMSuite:
    """Renvoie les sorties dans l'ordre et retient les consignes reçues."""

    def __init__(self, *sorties: str):
        self._sorties = list(sorties)
        self.consignes: list[str] = []

    def generer(self, consigne: str, question: str) -> str:
        self.consignes.append(consigne)
        return self._sorties.pop(0)


_OMISSION_1381 = json.dumps({
    "complete": False,
    "regle_generale": "Sont également soumis à la taxe foncière sur les propriétés bâties",
    "manque": "Les constructions sont soumises (art. 1381), sauf exonération de l'art. 1382.",
})


def test_completude_regle_generale_d_une_exoneration_remplie_ignoree():
    # v090 : la réponse applique l'exonération (1382) qui déroge à la règle
    # générale (1381) ; la situation la remplit → pas d'omission.
    llm = _LLMSuite(_OMISSION_1381, json.dumps({
        "phrase": "Les bâtiments qui servent aux exploitations rurales tels que granges, écuries",
        "remplie": True,
    }))
    assert verifier_completude("Ma grange agricole ?", _reponse_exoneree(), [_ART_1381, _ART_1382], llm) is None
    assert llm.consignes[1] == CONSIGNE_SITUATION_EXCEPTION


@pytest.mark.parametrize("verdict", [
    {"phrase": "Les bâtiments qui servent aux exploitations rurales", "remplie": False},
    {"phrase": "phrase absente des passages cités", "remplie": True},
    "pas du json",
])
def test_completude_exoneration_non_remplie_ou_sans_preuve_garde_l_omission(verdict):
    brut = verdict if isinstance(verdict, str) else json.dumps(verdict)
    llm = _LLMSuite(_OMISSION_1381, brut)
    manque = verifier_completude("Ma grange devenue gîte ?", _reponse_exoneree(), [_ART_1381, _ART_1382], llm)
    assert manque is not None and "1381" in manque


def test_completude_derogation_dans_le_meme_article_sans_controle_de_situation():
    # q58 : la règle générale omise est dans l'article cité lui-même → pas
    # d'exonération « d'ailleurs » : l'omission est gardée sans appel de plus.
    llm = _LLMSuite(_verdict_incomplet("dans la limite annuelle de 10 700 €"))
    assert verifier_completude("Q ?", _reponse_plafond_majore(), [_ARTICLE_156], llm) is not None
    assert len(llm.consignes) == 1


def test_exonerations_derogeant_exige_la_meme_section():
    autre = _Resultat("code_impot", _ART_1381.content, article="1381", section="Autre > A : Imposables")
    assert _exonerations_derogeant(_reponse_exoneree(), [autre, _ART_1382], [autre]) == []
    assert _exonerations_derogeant(_reponse_exoneree(), [_ART_1381, _ART_1382], [_ART_1381]) == [
        "Les bâtiments qui servent aux exploitations rurales tels que granges, écuries"
    ]
