"""Append-only change ledger for images managed locally after cold start.
A future sync job replays this to push deltas to S3."""

import json
from pathlib import Path

from . import config


def record(op: str, kind: str, name: str, ts: str) -> None:
    """op: add|replace|delete ; kind: original|rmbg ; ts: caller-supplied ISO."""
    path = Path(config.LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"op": op, "kind": kind, "name": name, "ts": ts})
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
