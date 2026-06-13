"""Tracing helpers for the search pipeline.

Goal: every step of /search is visible in `docker compose logs -f`,
so the user can see where time is being spent and confirm each stage
ran. `flush=True` because Docker stdout buffering hides progress
otherwise (we already set PYTHONUNBUFFERED=1 in the Dockerfile, this
is belt-and-suspenders).

Set env DEBUG=0 to silence (production deploy).
"""

import os
import sys
import time
from contextlib import contextmanager


ENABLED = os.environ.get("DEBUG", "1") != "0"


def dbg(msg: str) -> None:
    if ENABLED:
        print(f"[search] {msg}", flush=True)


@contextmanager
def timed(label: str):
    """Time a block; prints `  label: Xms` on exit."""
    if not ENABLED:
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        dt_ms = (time.time() - t0) * 1000.0
        print(f"[search]   {label}: {dt_ms:.0f}ms", flush=True)
