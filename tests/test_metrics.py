"""Tests des mesures de qualité de la recherche."""
from __future__ import annotations

from eval.metrics import hit_at_k, mrr, rang_premier_pertinent, taux_hit_at_k

A = ("loi_06_07_1989", "22")
B = ("loi_06_07_1989", "15")
C = ("Loi_hoguet", "3")


def test_rang_premier_pertinent_trouve_la_premiere_cible():
    assert rang_premier_pertinent([C, B, A], {A, B}) == 2


def test_rang_premier_pertinent_aucune_cible():
    assert rang_premier_pertinent([C, C, C], {A}) is None


def test_rang_premier_pertinent_liste_vide():
    assert rang_premier_pertinent([], {A}) is None


def test_hit_at_k():
    assert hit_at_k(3, 5) is True
    assert hit_at_k(5, 5) is True
    assert hit_at_k(6, 5) is False
    assert hit_at_k(None, 5) is False


def test_taux_hit_at_k_compte_la_proportion():
    # rangs : 1, 4, None -> dans le top 3 : seul le premier
    assert taux_hit_at_k([1, 4, None], 3) == 1 / 3
    assert taux_hit_at_k([1, 4, None], 5) == 2 / 3


def test_taux_hit_at_k_lot_vide():
    assert taux_hit_at_k([], 5) == 0.0


def test_mrr_moyenne_des_rangs_reciproques():
    # 1/1 + 1/2 + 0, sur 3 questions
    assert mrr([1, 2, None]) == (1.0 + 0.5 + 0.0) / 3


def test_mrr_parfait_et_nul():
    assert mrr([1, 1, 1]) == 1.0
    assert mrr([None, None]) == 0.0


def test_mrr_lot_vide():
    assert mrr([]) == 0.0
