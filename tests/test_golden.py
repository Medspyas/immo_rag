"""Tests du jeu de questions de référence.

Au-delà du simple chargement, ces tests confrontent chaque article attendu au
corpus réel (data/clean) : une coquille dans un numéro d'article est ainsi
repérée tout de suite, pas au moment de l'évaluation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import decouper_en_articles
from eval.golden import charger_golden, questions_autonomes

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_CORPUS = RACINE / "data" / "clean"


def test_le_golden_se_charge_et_compte_assez_de_cas():
    cas = charger_golden()
    assert len(cas) >= 20


def test_identifiants_uniques():
    ids = [c.id for c in charger_golden()]
    assert len(ids) == len(set(ids))


def test_presence_de_cas_hors_corpus():
    hors_corpus = [c for c in charger_golden() if c.hors_corpus]
    assert len(hors_corpus) >= 3


def test_cas_du_corpus_ont_une_cible_exploitable():
    for c in charger_golden():
        if c.hors_corpus:
            continue
        assert c.doc_id
        assert c.articles and all(a.strip() for a in c.articles)


def test_doc_id_correspondent_a_des_fichiers_du_corpus():
    if not DOSSIER_CORPUS.is_dir():
        pytest.skip("corpus data/clean absent")
    stems = {p.stem for p in DOSSIER_CORPUS.glob("*.txt")}
    for c in charger_golden():
        if c.hors_corpus:
            continue
        assert c.doc_id in stems, f"{c.id} : doc_id inconnu ({c.doc_id!r})"


def test_les_articles_cibles_existent_dans_le_corpus():
    if not DOSSIER_CORPUS.is_dir():
        pytest.skip("corpus data/clean absent")

    articles_par_doc: dict[str, set[str]] = {}
    for c in charger_golden():
        if c.hors_corpus:
            continue
        if c.doc_id not in articles_par_doc:
            texte = (DOSSIER_CORPUS / f"{c.doc_id}.txt").read_text(encoding="utf-8")
            articles_par_doc[c.doc_id] = {
                rc.article for rc in decouper_en_articles(texte, c.doc_id)
            }
        manquants = set(c.articles) - articles_par_doc[c.doc_id]
        assert not manquants, f"{c.id} : articles absents de {c.doc_id} : {sorted(manquants)}"


def test_presence_de_questions_de_suivi_et_exclusion_de_la_recherche_seule():
    tous = charger_golden()
    suivis = [c for c in tous if c.historique]
    assert len(suivis) >= 3
    assert len(questions_autonomes(tous)) == len(tous) - len(suivis)
