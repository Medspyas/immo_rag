"""Tests du développement des sigles (app/rag/sigles.py)."""
from __future__ import annotations

from app.rag.sigles import developper_sigles


def test_sigle_suivi_de_sa_forme_developpee():
    assert developper_sigles("Qui doit payer l'IFI ?") == (
        "Qui doit payer l'IFI (impôt sur la fortune immobilière) ?"
    )


def test_sigle_a_l_interieur_d_un_mot_ignore():
    assert developper_sigles("L'AGENT immobilier") == "L'AGENT immobilier"


def test_sigle_en_minuscules_ignore():
    assert developper_sigles("une ag extraordinaire") == "une ag extraordinaire"


def test_forme_deja_presente_non_repetee():
    question = "Le DPE (diagnostic de performance énergétique) est-il obligatoire ?"
    assert developper_sigles(question) == question


def test_question_sans_sigle_inchangee():
    question = "Quel est le délai de préavis du locataire ?"
    assert developper_sigles(question) == question
