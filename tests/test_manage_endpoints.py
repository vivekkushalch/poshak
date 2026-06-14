from fastapi.testclient import TestClient
from service import main, config
from service.store import connect


def test_startup_creates_job_tables(tmp_path, monkeypatch):
    db = tmp_path / "index.sqlite"
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(config, "SQLITE_PATH", db)
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    with TestClient(main.app):   # triggers lifespan startup
        pass
    with connect(str(db)) as c:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"images", "index_jobs", "index_job_items"} <= names


def test_ingest_stages_and_enqueues(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path / "uploads")
    captured = {}
    monkeypatch.setattr(main.jobs, "submit",
                        lambda *a, **k: captured.update({"args": a, "kw": k}))
    c = TestClient(main.app)
    r = c.post("/ingest",
               data={"sku_id": 5, "designno": "D", "item": "Fabric",
                     "photo_name": "z.jpg", "batch_id": "B9"},
               files={"image": ("z.jpg", b"x", "image/jpeg")})
    assert r.status_code == 202
    body = r.json()
    assert body["batch_id"] == "B9" and body["queued"] is True
    assert (tmp_path / "uploads" / "z.jpg").read_bytes() == b"x"
    assert captured["args"][0] == "B9" and captured["args"][1] == "z.jpg"


def test_ingest_generates_batch_id_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main.jobs, "submit", lambda *a, **k: None)
    c = TestClient(main.app)
    r = c.post("/ingest",
               data={"sku_id": 1, "item": "Sherwani", "photo_name": "a.jpg"},
               files={"image": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 202 and len(r.json()["batch_id"]) > 0


def test_ingest_rejects_bad_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path / "uploads")
    c = TestClient(main.app)
    r = c.post("/ingest",
               data={"sku_id": 1, "item": "Sherwani", "photo_name": "a.jpg",
                     "bg_engine": "bogus"},
               files={"image": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 400


def test_get_jobs_list_and_detail(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(main, "list_jobs", lambda db: [{"batch_id": "B1"}])
    monkeypatch.setattr(main, "get_job",
                        lambda db, bid: {"batch_id": bid, "status": "done"})
    c = TestClient(main.app)
    assert c.get("/index/jobs").json()["jobs"][0]["batch_id"] == "B1"
    assert c.get("/index/jobs/B1").json()["status"] == "done"


def test_get_job_404(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(main, "get_job", lambda db, bid: None)
    c = TestClient(main.app)
    assert c.get("/index/jobs/nope").status_code == 404


def test_forget_calls_forget(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(main, "forget_image",
                        lambda name: {"files_removed": 1, "db_deleted": 1})
    c = TestClient(main.app)
    r = c.post("/forget", data={"photo_name": "z.jpg"})
    assert r.status_code == 200 and r.json()["db_deleted"] == 1


def test_replace_rejects_bad_kind(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    c = TestClient(main.app)
    r = c.post("/replace", data={"photo_name": "z.jpg", "kind": "bogus"},
               files={"image": ("z.jpg", b"x", "image/jpeg")})
    assert r.status_code == 400
