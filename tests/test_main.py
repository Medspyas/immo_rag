import os

os.environ.setdefault("DATABASE_URL", "postgresql://immo:immo@localhost:5433/immo_rag")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_repond_200_meme_si_la_base_est_injoignable():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
