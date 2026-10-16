import json

from app.rag.suivi import ECHANGES_MAX, Echange, reformuler_question


class _LLM:
    def __init__(self, sortie):
        self.sortie = sortie
        self.messages = []

    def generer(self, consigne, question):
        self.messages.append(question)
        if isinstance(self.sortie, Exception):
            raise self.sortie
        return self.sortie


HISTORIQUE = [Echange(question="Quel préavis pour le locataire ?", reponse="Trois mois.")]


def test_sans_historique_aucun_appel():
    llm = _LLM(json.dumps({"question": "autre"}))
    assert reformuler_question("Quel préavis ?", [], llm) == "Quel préavis ?"
    assert llm.messages == []


def test_question_de_suivi_reecrite():
    llm = _LLM(json.dumps({"question": "Quel préavis pour un meublé ?"}))
    assert reformuler_question("Et pour un meublé ?", HISTORIQUE, llm) == "Quel préavis pour un meublé ?"
    assert "Quel préavis pour le locataire ?" in llm.messages[0]
    assert "Nouvelle question : Et pour un meublé ?" in llm.messages[0]


def test_echec_du_modele_garde_la_question():
    assert reformuler_question("Et pour un meublé ?", HISTORIQUE, _LLM(RuntimeError("quota"))) == (
        "Et pour un meublé ?"
    )


def test_sortie_illisible_ou_vide_garde_la_question():
    for sortie in ("pas du json", json.dumps({"question": "  "}), json.dumps({"autre": 1})):
        assert reformuler_question("Et alors ?", HISTORIQUE, _LLM(sortie)) == "Et alors ?"


def test_seuls_les_derniers_echanges_et_le_debut_des_reponses_sont_transmis():
    historique = [Echange(question=f"question {i}", reponse="x" * 5000) for i in range(5)]
    llm = _LLM(json.dumps({"question": "ok ?"}))
    reformuler_question("Et alors ?", historique, llm)
    message = llm.messages[0]
    assert "question 0" not in message
    assert f"question {5 - ECHANGES_MAX}" in message
    assert "x" * 801 not in message


def test_refus_precedent_signale_au_modele():
    llm = _LLM(json.dumps({"question": "ok ?"}))
    reformuler_question("Et alors ?", [Echange(question="Taxe ?")], llm)
    assert "refusé" in llm.messages[0]
