from fastapi.testclient import TestClient

from poshak import config, main


def test_ui_page_served(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    r = TestClient(main.app).get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "poshak image search" in r.text.lower()


def test_admin_page_served(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    r = TestClient(main.app).get("/admin")
    assert r.status_code == 200
    assert 'id="grid"' in r.text


def test_static_common_js_served():
    r = TestClient(main.app).get("/static/common.js")
    assert r.status_code == 200
    assert "apiFetch" in r.text


def test_ui_has_search_controls():
    r = TestClient(main.app).get("/ui")
    assert 'id="file"' in r.text
    assert 'capture="environment"' in r.text
    assert 'id="results"' in r.text
    assert "ui.js" in r.text


def test_admin_has_grid_and_pager():
    r = TestClient(main.app).get("/admin")
    assert 'id="grid"' in r.text
    assert 'id="pager"' in r.text
    assert "admin.js" in r.text
