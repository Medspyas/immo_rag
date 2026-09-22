import os

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
