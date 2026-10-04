"""RANKER=v2 through the real container, services and HTTP layer (models and disks faked)."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from poshak.core.config import Settings
from poshak.engine import sift
from poshak.main import create_app
from tests.fakes import make_container, png
from tests.golden.fixtures import catalog, queries


def build(tmp_path, **settings):
    c = make_container(tmp_path, **settings)
    for row in catalog():
        c.indexer.stage(row["photo"], png(row["image"]))
        item = "Sherwani" if row["category"] == "cloth" else "Fabric"
        c.indexer.index(row["photo"], row["sku_id"], "", item, reload=False)
    c.state.reload()
    return c


@pytest.fixture
def v2(tmp_path):
    return build(tmp_path / "v2", ranker="v2")


@pytest.mark.parametrize(
    ("query", "sku"),
    [("cloth_noisy", 2), ("cloth_crop", 4), ("fabric_noisy", 8), ("fabric_crop", 9)],
)
def test_known_queries_rank_their_sku_first(v2, query, sku):
    category, img = queries()[query]
    results = v2.search.search(png(img), category, top_k=3)
    assert results[0]["sku_id"] == sku
    assert results[0]["total_sift_matches"] >= 16


def test_cloth_results_have_the_api_shape_without_a_cutout_pass(v2):
    category, img = queries()["cloth_noisy"]
    top = v2.search.search(png(img), category, top_k=1)[0]
    assert top["bg_sift"] == 0 and top["bg_tier"] == "NONE" and top["both_found"] is False
    assert top["orig_sift"] == top["total_sift_matches"]
    assert top["best_entry"].json_filename == "2_0.jpg"


def test_v2_never_runs_background_removal_on_the_query(v2):
    for name in ("cloth_noisy", "cloth_crop", "cloth_unrelated"):
        category, img = queries()[name]
        v2.search.search(png(img), category)
    assert v2.search._query_remover.calls == 0


def test_the_embedding_is_computed_only_when_the_cheap_stages_are_not_decisive(v2):
    before = v2.embedder.calls
    category, img = queries()["cloth_noisy"]
    v2.search.search(png(img), category)
    assert v2.embedder.calls == before  # an easy query never reaches the embedding

    category, img = queries()["cloth_unrelated"]
    v2.search.search(png(img), category)
    assert v2.embedder.calls == before + 1  # a hard one embeds exactly once


def test_search_over_http_returns_the_documented_cloth_shape(v2):
    category, img = queries()["cloth_crop"]
    with TestClient(create_app(container=v2)) as client:  # the context runs the lifespan
        r = client.post(
            "/search", data={"category": category}, files={"image": ("q.png", png(img))}
        )
    body = r.json()
    assert r.status_code == 200 and body["category"] == "cloth"
    assert body["results"][0]["sku_id"] == 4
    assert body["results"][0]["best_entry"]["json_filename"] == "4_1.jpg"


def test_health_reports_the_ranker_and_stale_descriptors(v2, tmp_path):
    with TestClient(create_app(container=v2)) as client:
        body = client.get("/health").json()
    assert body["ranker"] == "v2"
    assert body["ranker_active"] == {"cloth": "v2", "fabric": "v2"}
    assert body["descriptors_to_refresh"] == 0

    with TestClient(create_app(container=build(tmp_path / "legacy"))) as legacy:
        body = legacy.get("/health").json()
    assert body["ranker"] == "legacy" and body["descriptors_to_refresh"] is None


def test_refresh_descriptors_upgrades_files_written_by_the_legacy_ranker(v2):
    store = v2.descriptors
    for key in list(store.data):  # downgrade every file to the legacy format
        des, n_kp = sift.loads(store.get(key))
        store.put(key, sift.dumps(des, n_kp))
    v2.state.reload()
    with TestClient(create_app(container=v2)) as client:
        stale = client.get("/health").json()["descriptors_to_refresh"]
        assert stale == sum(v2.state.current.stats()[k] for k in ("cloth_orig", "fabric")) > 0

        refreshed = v2.indexer.refresh_descriptors()
        assert refreshed["missing"] == 0 and refreshed["refreshed"] == len(store.data)
        assert client.get("/health").json()["descriptors_to_refresh"] == 0
    category, img = queries()["cloth_crop"]
    assert v2.search.search(png(img), category)[0]["sku_id"] == 4


def test_a_catalog_above_the_view_limit_falls_back_to_the_legacy_ranker(tmp_path):
    small = build(tmp_path / "small", ranker="v2", v2_max_views=3)
    legacy = build(tmp_path / "legacy")
    assert small.search.active_ranker("cloth") == "legacy"
    category, img = queries()["cloth_noisy"]
    got = small.search.search(png(img), category, top_k=3)
    want = legacy.search.search(png(img), category, top_k=3)
    assert [(r["sku_id"], r["total_sift_matches"]) for r in got] == [
        (r["sku_id"], r["total_sift_matches"]) for r in want
    ]


def test_ranker_setting_is_validated():
    assert Settings(disable_auth=True).ranker == "legacy"
    assert Settings(disable_auth=True, ranker="v2").v2_max_views == 1000
    with pytest.raises(ValidationError):
        Settings(disable_auth=True, ranker="fast")
    with pytest.raises(ValidationError):
        Settings(disable_auth=True, v2_max_views=0)
