import json

import pytest

from poshak import cli, container
from poshak.core import config
from tests.fakes import make_container, png
from tests.golden.fixtures import catalog


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """`poshak seed` against a container wired with fakes."""
    c = make_container(tmp_path)
    monkeypatch.setattr(container.Container, "build", classmethod(lambda cls, s: c))
    monkeypatch.setattr(config, "get_settings", lambda: c.settings)
    rows = catalog()[:2]  # sku 1, both photos
    c.originals.put(rows[0]["photo"], png(rows[0]["image"]))  # second photo is missing
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"skus": [
        {"stock_id": 1, "designno": "D1", "item": "Sherwani",
         "images_on_server": [r["photo"] for r in rows]},
    ]}))  # fmt: skip
    return c, path


def test_seed_indexes_present_photos_and_counts_missing(seeded, capsys):
    c, path = seeded
    assert cli.main(["seed", "--json", str(path)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        "done": 1, "skipped": 0, "missing": 1, "failed": 0,
    }  # fmt: skip
    assert c.indexer.is_indexed("1_0.jpg")


def test_seed_is_resumable(seeded, capsys):
    _, path = seeded
    cli.main(["seed", "--json", str(path)])
    cli.main(["seed", "--json", str(path)])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["skipped"] == 1


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])
