"""Tests du gate d'évaluation (eval/gate.py).

Seule la comparaison aux planchers est testée ici : la mesure elle-même
réutilise evaluer_systeme et recherche_hybride, déjà couverts par
test_ablation.py.
"""
from __future__ import annotations

import json

from eval.gate import SEUILS, intervalle, verifier_seuils


def test_tout_passe():
    mesures = {"mrr": 0.70, "hors_corpus_max": 0.13}
    seuils = {"mrr": {"min": 0.68}, "hors_corpus_max": {"max": 0.3}}
    assert verifier_seuils(mesures, seuils) == []


def test_plancher_franchi():
    echecs = verifier_seuils({"mrr": 0.60}, {"mrr": {"min": 0.68}})
    assert len(echecs) == 1
    assert "mrr" in echecs[0]


def test_plafond_franchi():
    echecs = verifier_seuils({"hors_corpus_max": 0.35}, {"hors_corpus_max": {"max": 0.3}})
    assert len(echecs) == 1
    assert "plafond" in echecs[0]


def test_egalite_au_plancher_passe():
    assert verifier_seuils({"mrr": 0.68}, {"mrr": {"min": 0.68}}) == []


def test_mesure_absente_est_un_echec():
    echecs = verifier_seuils({}, {"hit@5": {"min": 0.8}})
    assert echecs == ["hit@5 : mesure absente"]


def test_fichier_de_seuils_lisible():
    seuils = json.loads(SEUILS.read_text(encoding="utf-8"))
    assert seuils
    for borne in seuils.values():
        assert set(borne) <= {"min", "max"}


def test_intervalle_encadre_la_mesure():
    rangs = [1] * 28 + [2] * 6 + [None] * 3 + [4] * 3
    mrr = sum(1 / r for r in rangs if r) / len(rangs)
    bas, haut = intervalle(rangs, "mrr")
    assert bas < mrr < haut


def test_intervalle_nul_quand_toutes_les_questions_se_valent():
    assert intervalle([1] * 20, "hit@1") == (1.0, 1.0)
    assert intervalle([None] * 20, "hit@5") == (0.0, 0.0)


def test_intervalle_se_resserre_avec_plus_de_questions():
    petit = intervalle([1, None] * 20, "hit@1")
    grand = intervalle([1, None] * 100, "hit@1")
    assert grand[1] - grand[0] < petit[1] - petit[0]
