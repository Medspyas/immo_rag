import json

from app.rag.questions_articles import FICHIER_QUESTIONS, charger_questions


def test_charge_les_questions_par_loi_et_article(tmp_path):
    fichier = tmp_path / "q.jsonl"
    fichier.write_text(
        json.dumps({"doc_id": "hoguet", "article": "7", "questions": ["Durée du mandat ?"]}) + "\n",
        encoding="utf-8",
    )
    assert charger_questions.__wrapped__(fichier) == {("hoguet", "7"): ["Durée du mandat ?"]}


def test_fichier_absent_donne_un_dictionnaire_vide(tmp_path):
    assert charger_questions.__wrapped__(tmp_path / "absent.jsonl") == {}


def test_fichier_livre_couvre_les_trois_lois():
    lois = {doc_id for doc_id, _ in charger_questions.__wrapped__(FICHIER_QUESTIONS)}
    assert {"Loi_hoguet", "loi_06_07_1989", "loi-n°65-557_du_10_juillet_1965"} <= lois
