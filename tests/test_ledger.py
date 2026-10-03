import json

from poshak import ledger


def test_append_writes_one_jsonl_line_per_change(tmp_path, monkeypatch):
    p = tmp_path / "pending.jsonl"
    monkeypatch.setattr(ledger.config, "LEDGER_PATH", p)
    ledger.record("add", "original", "a.jpg", ts="2026-06-14T00:00:00")
    ledger.record("replace", "rmbg", "a.png", ts="2026-06-14T00:01:00")
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {"op": "add", "kind": "original", "name": "a.jpg", "ts": "2026-06-14T00:00:00"}
