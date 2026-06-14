import importlib


def test_auth_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    monkeypatch.setenv("DISABLE_AUTH", "0")
    cfg = importlib.reload(importlib.import_module("service.config"))
    assert cfg.API_KEY == "secret123"
    assert cfg.DISABLE_AUTH is False


def test_disable_auth_truthy(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    cfg = importlib.reload(importlib.import_module("service.config"))
    assert cfg.DISABLE_AUTH is True
