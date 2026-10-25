"""GET /api/system: training hardware and how MLflow is reached."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import gpu
from app.config import get_settings
from app.routers import runs


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(gpu, "detect", lambda: None)
    a = FastAPI()
    a.include_router(runs.router)
    yield TestClient(a)
    get_settings.cache_clear()


def test_system_reports_mlflow_mode(client, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000/mlflow")
    monkeypatch.setenv("MLFLOW_BROWSE_URL", "https://mlflow.example.com")
    get_settings.cache_clear()
    assert client.get("/api/system").json()["mlflow"] == {"enabled": True, "browseUrl": "https://mlflow.example.com"}
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    get_settings.cache_clear()
    assert client.get("/api/system").json()["mlflow"] == {"enabled": False, "browseUrl": None}
