"""Services wired through the real container with in-memory blobs and fake models."""

import json
import threading
from pathlib import Path

import pytest

from poshak.domain.errors import IndexNotReady, InvalidImage, NotFound, Overloaded
from poshak.domain.models import JobItem
from tests.fakes import FakeRemover, make_container, png
from tests.golden.fixtures import catalog, queries, summarize

GOLDEN = Path(__file__).parents[1] / "golden" / "search_golden.json"


@pytest.fixture
def c(tmp_path):
    container = make_container(tmp_path)
    yield container
    container.shutdown()


def _index_catalog(c):
    for row in catalog():
        item = "Sherwani" if row["category"] == "cloth" else "Fabric"
        c.indexer.stage(row["photo"], png(row["image"]))
        c.indexer.index(row["photo"], row["sku_id"], "", item, reload=False)
    c.state.reload()


def test_end_to_end_search_matches_golden(c):
    """Bytes in, ranked SKUs out, through indexer + index state + search service."""
    _index_catalog(c)
    got = {
        name: summarize(category, c.search.search(png(img), category, top_k=5))
        for name, (category, img) in queries().items()
    }
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_search_before_index_is_not_ready(c):
    with pytest.raises(IndexNotReady):
        c.search.search(png(catalog()[0]["image"]), "fabric")


def test_search_rejects_undecodable_bytes(c):
    _index_catalog(c)
    with pytest.raises(InvalidImage):
        c.search.search(b"not an image", "fabric")


def test_search_sheds_load_when_slots_are_full(tmp_path):
    c = make_container(tmp_path, max_concurrent_searches=1)
    c.search._queue_timeout_s = 0.05
    c.search._slots.acquire()  # simulate one search in flight
    try:
        with pytest.raises(Overloaded):
            c.search.search(b"x", "fabric")
    finally:
        c.search._slots.release()
        c.shutdown()


def test_index_is_idempotent_unless_forced(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    assert c.indexer.index(row["photo"], 1, "D", "Sherwani")["status"] == "indexed"
    assert c.indexer.index(row["photo"], 1, "D", "Sherwani")["status"] == "skipped"
    assert c.indexer.index(row["photo"], 1, "D", "Sherwani", force=True)["status"] == "indexed"


def test_cloth_gets_a_cutout_fabric_does_not(c):
    cloth, fabric = catalog()[0], catalog()[-1]
    for row, item in ((cloth, "Sherwani"), (fabric, "Fabric")):
        c.indexer.stage(row["photo"], png(row["image"]))
        c.indexer.index(row["photo"], row["sku_id"], "", item)
    assert c.cutouts.exists("1_0.png") and not c.cutouts.exists("10_1.png")
    assert c.state.current.stats() == {"cloth_bg": 1, "cloth_orig": 1, "fabric": 1}


def test_reuse_cutout_skips_background_removal(tmp_path):
    c = make_container(tmp_path)
    remover = FakeRemover()
    c.indexer._remover = remover
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    c.cutouts.put("1_0.png", png(row["image"]))
    res = c.indexer.index(row["photo"], 1, "", "Sherwani", reuse_cutout=True)
    assert res["bg_engine"] is None and remover.calls == 0
    c.shutdown()


def test_unknown_item_type_is_inferred_and_remembered(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    assert c.indexer.index(row["photo"], 1, "", "Linen Fabric")["category"] == "fabric"
    assert {"item": "Linen Fabric", "category": "fabric", "source": "inferred"} in (
        c.catalog.categories()
    )


def test_manual_category_survives_reseeding(c):
    c.catalog.set_category("Sherwani", "fabric")
    c.catalog.seed_defaults(["Sherwani"], [])
    assert c.catalog.resolve("Sherwani") == "fabric"


def test_replace_cutout_and_original(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    c.indexer.index(row["photo"], 1, "", "Sherwani")
    other = png(catalog()[1]["image"])
    assert c.indexer.replace(row["photo"], "rmbg", other)["status"] == "replaced"
    assert c.cutouts.get("1_0.png") == other
    c.indexer.replace(row["photo"], "original", other)
    assert c.originals.get(row["photo"]) == other


def test_replace_unknown_photo_is_not_found(c):
    with pytest.raises(NotFound):
        c.indexer.replace("ghost.jpg", "original", b"x")


def test_forget_removes_files_row_and_index_entry(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    c.indexer.index(row["photo"], 1, "", "Sherwani")
    assert c.indexer.forget(row["photo"]) == {"files_removed": 4, "db_deleted": 1}
    assert not c.state.current.ready


def test_jobs_record_success_failure_and_resume(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    c.jobs.submit(JobItem("B1", row["photo"], 1, "", "Sherwani"))
    c.jobs.submit(JobItem("B1", "missing.jpg", 2, "", "Sherwani"))
    c.jobs.run_until_empty()
    job = c.jobs._jobs.get("B1")
    assert (job["status"], job["done"], job["failed"], job["local_count"]) == ("done", 1, 1, 1)
    assert job["errors"][0]["photo_name"] == "missing.jpg"
    assert c.state.current.stats()["cloth_orig"] == 1

    c.jobs._jobs.add_item(JobItem("B2", "later.jpg", 3, "", "Fabric"))  # left queued
    assert c.jobs.resume() == 1


def test_reload_swaps_snapshot_atomically(c):
    before = c.state.current
    _index_catalog(c)
    after = c.state.current
    assert before is not after and not before.ready and after.ready


def test_background_worker_drains_queue(c):
    row = catalog()[0]
    c.indexer.stage(row["photo"], png(row["image"]))
    done = threading.Event()
    original_reload = c.state.reload

    def reload_and_signal():
        stats = original_reload()
        done.set()
        return stats

    c.state.reload = reload_and_signal
    c.jobs.start()
    c.jobs.submit(JobItem("B1", row["photo"], 1, "", "Sherwani"))
    assert done.wait(10)
    assert c.jobs._jobs.get("B1")["done"] == 1
