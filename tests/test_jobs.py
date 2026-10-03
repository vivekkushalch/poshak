from poshak import jobs
from poshak.store import get_job, init_db, list_pending_items


def _wire(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
    monkeypatch.setattr(jobs.config, "SQLITE_PATH", db)
    monkeypatch.setattr(jobs, "_reload_store", lambda: None)
    return db


def test_submit_persists_and_enqueues(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    jobs.submit("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    assert len(list_pending_items(db)) == 1
    assert jobs._Q.qsize() == 1


def test_process_one_marks_done_and_counts_engine(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(
        jobs, "index_one", lambda *a, **k: {"status": "indexed", "bg_engine": "api"}
    )
    jobs.submit("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    jobs._process_one("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    job = get_job(db, "B1")
    assert job["done"] == 1 and job["api_count"] == 1 and job["status"] == "done"


def test_process_one_records_failure(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("embed exploded")

    monkeypatch.setattr(jobs, "index_one", boom)
    jobs.submit("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    jobs._process_one("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    job = get_job(db, "B1")
    assert job["failed"] == 1
    assert "embed exploded" in job["errors"][0]["error"]


def test_resume_reenqueues_pending(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    jobs.submit("B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    while not jobs._Q.empty():
        jobs._Q.get_nowait()
    jobs.resume()
    assert jobs._Q.qsize() == 1
