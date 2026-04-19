"""
Tests fonctionnels de l'API FastAPI.

Utilise TestClient (starlette) pour exécuter l'app en interne — pas de serveur séparé requis.
La couverture du code api/ et src/ est ainsi capturée par pytest-cov.

Usage : pytest tests/api_test.py -v --cov=api --cov=src --cov-report=html
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.main import app

client = TestClient(app, follow_redirects=False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["vectors"] > 0


def test_root_redirects_to_docs():
    r = client.get("/")
    assert r.status_code in (301, 302, 307, 308)
    assert "/docs" in r.headers["location"]


def test_ask_valid_question():
    r = client.post("/ask", json={"question": "Quels concerts sont prévus à Rennes ?"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 0
    assert len(data["contexts"]) > 0


def test_ask_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_whitespace_question():
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 422


def test_ask_missing_field():
    r = client.post("/ask", json={})
    assert r.status_code == 422


def test_rebuild():
    r = client.post("/rebuild")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["vectors"] > 0