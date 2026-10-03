import sqlite3

import numpy as np
import pytest

from poshak.domain.models import ImageRecord, JobItem
from poshak.infrastructure.db.migrations import migrate
from poshak.infrastructure.db.sqlite import (
    SqliteCategoryRepository,
    SqliteDatabase,
    SqliteImageRepository,
    SqliteJobRepository,
)


@pytest.fixture
def db(tmp_path):
    d = SqliteDatabase(tmp_path / "i.sqlite")
    d.init()
    return d


def _rec(photo="a.jpg", sku=1, tag="m", cut=True):
    v = np.arange(4, dtype=np.float32)
    return ImageRecord(
        sku_id=sku,
        designno="D",
        item="Sherwani",
        category="cloth",
        photo_name=photo,
        orig_vec=v,
        orig_desc=f"{photo}.orig.npz",
        model_tag=tag,
        rembg_name="a.png" if cut else None,
        rembg_vec=v * 2 if cut else None,
        rembg_desc=f"{photo}.rmbg.npz" if cut else None,
        bg_engine="local",
    )


def test_image_roundtrip_and_upsert(db):
    repo = SqliteImageRepository(db)
    repo.upsert(_rec())
    got = repo.get("a.jpg", "m")
    assert got.sku_id == 1 and np.array_equal(got.rembg_vec, np.arange(4) * 2)
    repo.upsert(_rec(sku=9))
    assert repo.get("a.jpg", "m").sku_id == 9
    assert len(repo.all("m")) == 1


def test_record_without_cutout_has_one_view(db):
    repo = SqliteImageRepository(db)
    repo.upsert(_rec(cut=False))
    orig, cut = repo.get("a.jpg", "m").views()
    assert orig.desc_key == "a.jpg.orig.npz" and cut is None


def test_model_tag_isolates_rows(db):
    repo = SqliteImageRepository(db)
    repo.upsert(_rec(tag="old"))
    assert repo.all("new") == [] and repo.get("a.jpg", "new") is None


def test_page_and_delete(db):
    repo = SqliteImageRepository(db)
    for i in range(5):
        repo.upsert(_rec(photo=f"{i}.jpg", sku=i))
    total, rows = repo.page("m", page=2, page_size=2)
    assert total == 5 and [r["photo_name"] for r in rows] == ["2.jpg", "3.jpg"]
    assert repo.delete("0.jpg", "m") == 1 and repo.delete("0.jpg", "m") == 0


def test_job_lifecycle(db):
    jobs = SqliteJobRepository(db)
    jobs.add_item(JobItem("B1", "a.jpg", 1, "D", "Sherwani"))
    jobs.add_item(JobItem("B1", "b.jpg", 2, "D", "Fabric", force=True))
    assert [j.photo_name for j in jobs.pending()] == ["a.jpg", "b.jpg"]
    jobs.set_status("B1", "a.jpg", "done")
    jobs.set_status("B1", "b.jpg", "failed", "boom")
    jobs.bump_engine("B1", "api")
    jobs.refresh_counts("B1")
    job = jobs.get("B1")
    assert (job["status"], job["done"], job["failed"], job["api_count"]) == ("done", 1, 1, 1)
    assert job["errors"] == [{"photo_name": "b.jpg", "error": "boom"}]
    assert jobs.pending() == [] and jobs.get("nope") is None


def test_categories(db):
    cats = SqliteCategoryRepository(db)
    cats.add("Sherwani", "cloth", "config")
    cats.add("Sherwani", "fabric", "config")  # insert-or-ignore keeps the first
    assert cats.get("sherwani") == "cloth"  # case-insensitive
    cats.set_manual("Sherwani", "fabric")
    assert cats.list() == [{"item": "Sherwani", "category": "fabric", "source": "manual"}]
    assert cats.get("Kurta") is None


def test_migration_turns_paths_into_keys_and_is_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO images (sku_id, category, photo_name, orig_vec, orig_desc, rembg_desc, "
        "model_tag) VALUES (1, 'cloth', 'a.jpg', x'00', ?, ?, 'm')",
        ("/data/descriptors/a.jpg.orig.npz", "C:\\data\\a.jpg.rmbg.npz"),
    )
    migrate(conn)
    migrate(conn)
    row = conn.execute("SELECT orig_desc, rembg_desc FROM images").fetchone()
    assert row == ("a.jpg.orig.npz", "a.jpg.rmbg.npz")
