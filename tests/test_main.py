import os

import psycopg

os.environ.setdefault("DATABASE_URL", "postgresql://immo:immo@localhost:5433/immo_rag")

from fastapi.testclient import TestClient

from app.main import app, settings

client = TestClient(app)

# DSN volontairement injoignable (port fermé) pour les deux tests "base
# injoignable" ci-dessous : dépendre de l'absence de Docker sur la machine de
# test cassait dès que la base tournait réellement (ex. pour une ingestion en
# conditions réelles) — corrigé le 2026-09-21.
_DSN_INJOIGNABLE = "postgresql://x:x@localhost:1/x"


def test_health_repond_200_meme_si_la_base_est_injoignable():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_renvoie_503_si_la_base_est_injoignable(monkeypatch):
    monkeypatch.setattr(settings, "database_url", _DSN_INJOIGNABLE)
    response = client.post("/ingest", json={"doc_id": "loi-test", "texte": "Article 1\nTexte."})
    assert response.status_code == 503


def test_query_renvoie_503_si_la_base_est_injoignable(monkeypatch):
    monkeypatch.setattr(settings, "database_url", _DSN_INJOIGNABLE)
    response = client.post("/query", json={"question": "Quel préavis ?"})
    assert response.status_code == 503


def test_ingest_refuse_un_texte_vide():
    response = client.post("/ingest", json={"doc_id": "loi-test", "texte": ""})
    assert response.status_code == 422


def test_query_refuse_une_question_vide():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_renvoie_503_si_le_modele_de_langage_est_sature(monkeypatch):
    from app import main
    from app.rag.generator import ModeleIndisponible

    def _sature(*args, **kwargs):
        raise ModeleIndisponible("quota atteint")

    monkeypatch.setattr(main, "_retriever", lambda: object())
    monkeypatch.setattr(main, "_llm", lambda: object())
    monkeypatch.setattr(main, "_juge", lambda: object())
    monkeypatch.setattr(main, "_nli", lambda: object())
    monkeypatch.setattr(main, "repondre", _sature)
    reponse = client.post("/query", json={"question": "Quel préavis ?"})
    assert reponse.status_code == 503


def test_query_refuse_une_question_trop_longue():
    from app.models import QUESTION_MAX_CARACTERES

    response = client.post("/query", json={"question": "a" * (QUESTION_MAX_CARACTERES + 1)})
    assert response.status_code == 422


class _BaseFactice:
    """Tient lieu de VectorStore pour les routes /documents."""

    def __init__(self, supprimes: int = 0):
        self.supprimes = supprimes
        self.supprime: list[str] = []

    def documents(self):
        from datetime import UTC, datetime

        from app.rag.vector_store import DocumentIndexe

        return [
            DocumentIndexe(
                doc_id="loi_06_07_1989", titre="Loi n° 89-462", morceaux=120,
                mis_a_jour=datetime(2026, 10, 5, tzinfo=UTC),
            )
        ]

    def delete_by_doc(self, doc_id: str) -> int:
        self.supprime.append(doc_id)
        return self.supprimes


def test_documents_liste_les_textes_indexes(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "_vector_store", lambda: _BaseFactice())
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()[0]["doc_id"] == "loi_06_07_1989"
    assert response.json()[0]["morceaux"] == 120


def test_documents_renvoie_503_si_la_base_est_injoignable(monkeypatch):
    from app import main

    def _injoignable():
        raise psycopg.OperationalError("base injoignable")

    monkeypatch.setattr(main, "_vector_store", _injoignable)
    assert client.get("/documents").status_code == 503


def test_supprimer_un_document_invalide_la_recherche(monkeypatch):
    from app import main

    base = _BaseFactice(supprimes=3)
    monkeypatch.setattr(main, "_vector_store", lambda: base)
    monkeypatch.setitem(main._cache, "bm25", object())
    monkeypatch.setitem(main._cache, "retriever", object())

    response = client.delete("/documents/loi_06_07_1989")

    assert response.status_code == 204
    assert base.supprime == ["loi_06_07_1989"]
    assert "bm25" not in main._cache and "retriever" not in main._cache


def test_supprimer_un_document_inconnu_renvoie_404(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "_vector_store", lambda: _BaseFactice(supprimes=0))
    assert client.delete("/documents/inconnu").status_code == 404


def test_query_transmet_l_historique(monkeypatch):
    from app import main
    from app.rag.generator import ReponseGeneree

    recu = {}

    def _repondre(question, *args, historique=None, **kwargs):
        recu["historique"] = historique
        return ReponseGeneree(refus=True, motif="test")

    monkeypatch.setattr(main, "_retriever", lambda: object())
    monkeypatch.setattr(main, "_llm", lambda: object())
    monkeypatch.setattr(main, "_juge", lambda: object())
    monkeypatch.setattr(main, "_nli", lambda: object())
    monkeypatch.setattr(main, "repondre", _repondre)
    historique = [{"question": "Quel préavis ?", "reponse": "Trois mois."}]
    reponse = client.post("/query", json={"question": "Et pour un meublé ?", "historique": historique})
    assert reponse.status_code == 200
    assert recu["historique"][0].question == "Quel préavis ?"


def test_query_refuse_un_historique_trop_long():
    from app.rag.suivi import ECHANGES_MAX

    historique = [{"question": "q", "reponse": ""}] * (ECHANGES_MAX + 1)
    response = client.post("/query", json={"question": "Et alors ?", "historique": historique})
    assert response.status_code == 422
