import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from service import auth, config


def _app():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(auth.require_api_key)])
    def protected():
        return {"ok": True}
    return TestClient(app, raise_server_exceptions=True)


def test_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "k")
    monkeypatch.setattr(config, "DISABLE_AUTH", False)
    r = _app().get("/protected")
    assert r.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "k")
    monkeypatch.setattr(config, "DISABLE_AUTH", False)
    r = _app().get("/protected", headers={"X-API-Key": "k"})
    assert r.status_code == 200


def test_bypass_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    r = _app().get("/protected")
    assert r.status_code == 200
