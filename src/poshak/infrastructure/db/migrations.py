"""SQLite schema and idempotent migrations. Safe to run on every startup."""

import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id         INTEGER PRIMARY KEY,
  sku_id     INTEGER NOT NULL,
  designno   TEXT,
  item       TEXT,
  category   TEXT NOT NULL,
  photo_name TEXT NOT NULL,
  rembg_name TEXT,
  orig_vec   BLOB NOT NULL,
  rembg_vec  BLOB,
  orig_desc  TEXT,
  rembg_desc TEXT,
  model_tag  TEXT NOT NULL,
  bg_engine  TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(photo_name, model_tag)
);
CREATE INDEX IF NOT EXISTS idx_model ON images(model_tag);
CREATE TABLE IF NOT EXISTS index_jobs (
  batch_id    TEXT PRIMARY KEY,
  status      TEXT NOT NULL DEFAULT 'open',
  total       INTEGER NOT NULL DEFAULT 0,
  done        INTEGER NOT NULL DEFAULT 0,
  failed      INTEGER NOT NULL DEFAULT 0,
  skipped     INTEGER NOT NULL DEFAULT 0,
  api_count   INTEGER NOT NULL DEFAULT 0,
  local_count INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS index_job_items (
  id         INTEGER PRIMARY KEY,
  batch_id   TEXT NOT NULL,
  photo_name TEXT NOT NULL,
  sku_id     INTEGER NOT NULL,
  designno   TEXT,
  item       TEXT,
  bg_engine  TEXT,
  force      INTEGER NOT NULL DEFAULT 0,
  status     TEXT NOT NULL DEFAULT 'queued',
  error      TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(batch_id, photo_name)
);
CREATE TABLE IF NOT EXISTS item_categories (
  item       TEXT PRIMARY KEY,
  category   TEXT NOT NULL,
  source     TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(images)")}
    if "bg_engine" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN bg_engine TEXT")
    _descriptor_paths_to_keys(conn)
    conn.commit()


def _descriptor_paths_to_keys(conn: sqlite3.Connection) -> None:
    """Older builds stored absolute descriptor paths; blob stores use bare keys."""
    rows = conn.execute(
        # char(47) = '/', char(92) = backslash
        "SELECT id, orig_desc, rembg_desc FROM images "
        "WHERE instr(orig_desc, char(47)) OR instr(orig_desc, char(92)) "
        "OR instr(rembg_desc, char(47)) OR instr(rembg_desc, char(92))"
    ).fetchall()
    for row_id, orig, rembg in rows:
        conn.execute(
            "UPDATE images SET orig_desc=?, rembg_desc=? WHERE id=?",
            (_basename(orig), _basename(rembg), row_id),
        )


def _basename(path: str | None) -> str | None:
    return None if path is None else os.path.basename(path.replace("\\", "/"))
