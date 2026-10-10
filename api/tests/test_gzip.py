"""Response compression: API JSON is gzipped, export and model downloads pass through untouched."""
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from app.main import JsonGZip

BODY = "x" * 5000


def _client() -> TestClient:
    a = FastAPI()
    a.add_middleware(JsonGZip, minimum_size=100, compresslevel=5)
    for path in ("/api/exports/3/files/data", "/api/runs/12/files/model", "/api/datasets"):
        a.add_api_route(path, lambda: PlainTextResponse(BODY))
    return TestClient(a)


def test_downloads_are_not_recompressed():
    c = _client()
    for path in ("/api/exports/3/files/data", "/api/runs/12/files/model"):
        r = c.get(path, headers={"accept-encoding": "gzip"})
        assert "content-encoding" not in r.headers and r.headers["content-length"] == str(len(BODY))


def test_api_responses_are_gzipped():
    r = _client().get("/api/datasets", headers={"accept-encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip" and r.text == BODY
