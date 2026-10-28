"""HTTP contract tests against the real app factory, wired with fakes."""

import pytest
from fastapi.testclient import TestClient

from poshak.main import create_app
from tests.fakes import make_container, png
from tests.golden.fixtures import catalog

KEY = {"X-API-Key": "s3cret"}


@pytest.fixture
def container(tmp_path):
    return make_container(tmp_path)


@pytest.fixture
def client(container):
    with TestClient(create_app(container=container)) as c:
        yield c


def _ingest(client, container, rows, headers=None):
    for row in rows:
        item = "Sherwani" if row["category"] == "cloth" else "Fabric"
        r = client.post(
            "/ingest",
            data={"sku_id": row["sku_id"], "item": item, "photo_name": row["photo"],
                  "batch_id": "B1"},
            files={"image": (row["photo"], png(row["image"]), "image/png")},
            headers=headers,
        )  # fmt: skip
        assert r.status_code == 202, r.text
    container.jobs.stop()  # deterministic: drain on this thread instead of the worker
    container.jobs.run_until_empty()


# --- system -------------------------------------------------------------------
def test_health_and_probes(client, container):
    body = client.get("/health").json()
    assert body["model_tag"] == container.settings.model_tag and body["index_ready"] is False
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    _ingest(client, container, catalog()[:1])
    assert client.get("/health/ready").status_code == 200


def test_config(tmp_path):
    c = make_container(tmp_path, image_base_url="https://img.test")
    with TestClient(create_app(container=c)) as client:
        assert client.get("/config").json()["image_base_url"] == "https://img.test"


def test_pages_and_static(client):
    ui = client.get("/ui")
    assert ui.status_code == 200 and "text/html" in ui.headers["content-type"]
    assert all(s in ui.text for s in ('id="file"', 'capture="environment"', 'id="results"'))
    admin = client.get("/admin").text
    assert 'id="grid"' in admin and 'id="pager"' in admin
    assert "apiFetch" in client.get("/static/common.js").text


def test_admin_app_is_complete_and_every_module_is_served(client):
    html = client.get("/admin").text
    assert all(f'data-panel="{tab}"' in html for tab in ("catalog", "upload", "jobs", "categories"))
    assert 'href="/static/admin.css"' in html and 'src="/static/admin/app.js"' in html
    for module in ("app", "kit", "api", "catalog", "upload", "jobs", "categories"):
        r = client.get(f"/static/admin/{module}.js")
        assert r.status_code == 200, module
    # every relative import inside the admin modules resolves to a served file
    for module in ("app", "catalog", "upload", "jobs", "categories"):
        for line in client.get(f"/static/admin/{module}.js").text.splitlines():
            if line.startswith("import") and '"./' in line:
                target = line.split('"./')[1].split('"')[0]
                assert client.get(f"/static/admin/{target}").status_code == 200, (module, target)


def test_static_files_revalidate_so_upgrades_are_never_stale(client):
    for path in ("/static/admin.css", "/static/admin/app.js", "/static/common.js"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    etag = client.get("/static/admin.css").headers["etag"]
    assert client.get("/static/admin.css", headers={"If-None-Match": etag}).status_code == 304


def test_missing_image_error_names_the_file(client):
    r = client.get("/img/original/ghost.jpg")
    assert r.status_code == 404 and r.json()["detail"] == "ghost.jpg not found"


def test_openapi_documents_both_result_shapes(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"ClothResult", "FabricResult", "ClothSearchResponse"} <= schemas.keys()


# --- auth ---------------------------------------------------------------------
def test_api_key_required_when_auth_enabled(tmp_path):
    c = make_container(tmp_path, disable_auth=False, api_key="s3cret")
    with TestClient(create_app(container=c)) as client:
        assert client.get("/skus").status_code == 401
        assert client.get("/skus", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/skus", headers=KEY).status_code == 200
        assert client.get("/health").status_code == 200  # probes stay open
        assert client.get("/img/original/x.jpg").status_code == 404  # images stay public


# --- search -------------------------------------------------------------------
def test_search_returns_ranked_cloth_results(client, container):
    rows = catalog()
    _ingest(client, container, rows)
    query = rows[2]["image"]  # sku 2, first photo
    r = client.post("/search", data={"category": "cloth", "top_k": 3},
                    files={"image": ("q.png", png(query), "image/png")})  # fmt: skip
    body = r.json()
    assert r.status_code == 200 and body["status"] == "success" and body["category"] == "cloth"
    top = body["results"][0]
    assert top["sku_id"] == 2 and top["both_found"] is True
    assert top["best_entry"]["json_filename"] == "2_0.jpg" == top["best_entry"]["filename"]
    assert len(body["results"]) == body["total_found"] == 3


def test_search_fabric_shape(client, container):
    rows = [r for r in catalog() if r["category"] == "fabric"]
    _ingest(client, container, rows)
    r = client.post("/search", data={"category": "fabric"},
                    files={"image": ("q.png", png(rows[0]["image"]), "image/png")})  # fmt: skip
    top = r.json()["results"][0]
    assert top["sku_id"] == rows[0]["sku_id"] and "tier" in top and "bg_sift" not in top


@pytest.mark.parametrize(
    ("data", "status"),
    [({"category": "bogus"}, 422), ({"category": "fabric", "top_k": 0}, 422)],
)
def test_search_validates_input(client, data, status):
    r = client.post("/search", data=data, files={"image": ("q.jpg", b"x", "image/jpeg")})
    assert r.status_code == status


def test_search_before_any_ingest_is_503(client):
    query = png(catalog()[0]["image"])
    r = client.post("/search", data={"category": "fabric"}, files={"image": ("q.png", query)})
    assert r.status_code == 503


def test_search_undecodable_image_is_400(client, container):
    _ingest(client, container, catalog()[:1])
    r = client.post("/search", data={"category": "cloth"},
                    files={"image": ("q.jpg", b"garbage", "image/jpeg")})  # fmt: skip
    assert r.status_code == 400 and "decode" in r.json()["detail"]


# --- ingest / jobs ------------------------------------------------------------
def test_ingest_queues_and_job_completes(client, container):
    _ingest(client, container, catalog()[:2])
    job = client.get("/index/jobs/B1").json()
    assert (job["status"], job["total"], job["done"], job["errors"]) == ("done", 2, 2, [])
    assert [j["batch_id"] for j in client.get("/index/jobs").json()["jobs"]] == ["B1"]


def test_ingest_generates_batch_id(client):
    r = client.post("/ingest", data={"sku_id": 1, "item": "Sherwani", "photo_name": "a.jpg"},
                    files={"image": ("a.jpg", b"x", "image/jpeg")})  # fmt: skip
    assert r.status_code == 202 and len(r.json()["batch_id"]) == 32


@pytest.mark.parametrize(
    "fields",
    [{"bg_engine": "gpu"}, {"photo_name": "../evil.jpg"}, {"photo_name": "a/b.jpg"}],
)
def test_ingest_validates_input(client, fields):
    data = {"sku_id": 1, "item": "Sherwani", "photo_name": "a.jpg", **fields}
    r = client.post("/ingest", data=data, files={"image": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/index/jobs/nope").status_code == 404


# --- replace / forget / images ------------------------------------------------
def test_replace_and_forget(client, container):
    row = catalog()[0]
    _ingest(client, container, [row])
    new = png(catalog()[1]["image"])
    r = client.post("/replace", data={"photo_name": row["photo"], "kind": "rmbg"},
                    files={"image": ("n.png", new, "image/png")})  # fmt: skip
    assert r.json() == {"status": "replaced", "photo_name": row["photo"], "kind": "rmbg"}
    img = client.get("/img/rmbg/1_0.png")
    assert img.content == new and img.headers["content-type"] == "image/png"

    r = client.post("/forget", data={"photo_name": row["photo"]})
    assert r.json() == {"status": "ok", "photo_name": row["photo"], "files_removed": 4,
                        "db_deleted": 1}  # fmt: skip
    assert client.get(f"/img/original/{row['photo']}").status_code == 404


def test_replace_validates(client):
    files = {"image": ("n.png", b"x", "image/png")}
    assert client.post("/replace", data={"photo_name": "a.jpg", "kind": "x"},
                       files=files).status_code == 422  # fmt: skip
    assert client.post("/replace", data={"photo_name": "ghost.jpg", "kind": "original"},
                       files=files).status_code == 404  # fmt: skip


# --- catalog ------------------------------------------------------------------
def test_items_paginate(client, container):
    _ingest(client, container, catalog()[:5])
    body = client.get("/skus", params={"page": 2, "page_size": 2}).json()
    assert body["total"] == 5 and [i["photo_name"] for i in body["items"]] == ["2_0.jpg", "2_1.jpg"]
    assert client.get("/skus", params={"page_size": 500}).status_code == 422


def test_categories_roundtrip(client):
    items = {c["item"] for c in client.get("/categories").json()["categories"]}
    assert {"Sherwani", "Fabric"} <= items
    r = client.post("/categories", data={"item": " Kurta ", "category": "fabric"})
    assert r.json() == {"status": "ok", "item": "Kurta", "category": "fabric"}
    assert client.post("/categories", data={"item": "K", "category": "x"}).status_code == 422


# --- demo mode: read-only, sample queries, landing redirect --------------------------
def test_root_redirects_to_the_search_page(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui"


def test_read_only_mode_allows_search_and_blocks_every_write(tmp_path):
    c = make_container(tmp_path, read_only=True)
    for row in catalog():
        item = "Sherwani" if row["category"] == "cloth" else "Fabric"
        c.indexer.stage(row["photo"], png(row["image"]))
        c.indexer.index(row["photo"], row["sku_id"], "", item, reload=False)
    c.state.reload()
    with TestClient(create_app(container=c)) as client:
        img = png(catalog()[0]["image"])
        ok = client.post("/search", data={"category": "cloth"}, files={"image": ("q.png", img)})
        assert ok.status_code == 200 and ok.json()["results"][0]["sku_id"] == 1
        assert client.get("/health").status_code == 200
        for path, kwargs in (
            (
                "/ingest",
                {
                    "data": {"sku_id": 9, "item": "Fabric", "photo_name": "x.jpg"},
                    "files": {"image": ("x.jpg", img)},
                },
            ),
            ("/forget", {"data": {"photo_name": "1_0.jpg"}}),
            ("/categories", {"data": {"item": "Kurta", "category": "cloth"}}),
        ):
            r = client.post(path, **kwargs)
            assert r.status_code == 403 and "read-only" in r.json()["detail"], path
        assert c.state.current.stats()["cloth_orig"] > 0  # nothing was deleted


def test_samples_are_served_from_a_flat_folder_only(tmp_path):
    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "index.json").write_text('[{"file": "a.jpg"}]', encoding="utf-8")
    (folder / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    with TestClient(create_app(container=make_container(tmp_path / "d", samples_dir=folder))) as c:
        assert c.get("/samples/index.json").json() == [{"file": "a.jpg"}]
        assert c.get("/samples/a.jpg").status_code == 200
        assert c.get("/samples/missing.jpg").status_code == 404
        assert c.get("/samples/..%2Fsecret.txt").status_code == 404
    with TestClient(create_app(container=make_container(tmp_path / "e"))) as c:
        assert c.get("/samples/index.json").status_code == 404  # no SAMPLES_DIR configured
