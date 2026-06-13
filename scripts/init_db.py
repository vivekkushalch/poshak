"""Idempotent schema bootstrap.

Replaces "run mysql.exe by hand". Detects three states and does the
right thing for each:

  1. No `stock_image_vectors` table at all
        → CREATE the new table.
  2. Old 512-d schema present (column `vec` exists, `orig_vec` doesn't)
        → DROP, then CREATE. Old vectors are unusable under the new
          model anyway. Prints a clear warning before dropping.
  3. New schema already present
        → No-op.

Run:
    docker compose exec image-search python -m scripts.init_db
    docker compose exec image-search python -m scripts.init_db --force  # always DROP+CREATE
"""

import argparse
from contextlib import closing
from pathlib import Path

from service.db import get_connection


SCHEMA_FILE = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

DETECT_TABLE_SQL = """
SELECT COUNT(*) AS c
  FROM information_schema.tables
 WHERE table_schema = DATABASE()
   AND table_name = 'stock_image_vectors'
"""

DETECT_COLUMN_SQL = """
SELECT column_name
  FROM information_schema.columns
 WHERE table_schema = DATABASE()
   AND table_name = 'stock_image_vectors'
"""


def _statements_from(sql: str):
    """Split a SQL file into statements. Strips comments. Naive — fine
    for our hand-written schema; no triggers or DELIMITER changes."""
    out = []
    for raw in sql.split(";"):
        cleaned = "\n".join(
            ln for ln in raw.splitlines() if not ln.strip().startswith("--")
        ).strip()
        if cleaned:
            out.append(cleaned)
    return out


def detect_state(cur) -> str:
    cur.execute(DETECT_TABLE_SQL)
    if cur.fetchone()["c"] == 0:
        return "missing"
    cur.execute(DETECT_COLUMN_SQL)
    cols = {r["column_name"].lower() for r in cur.fetchall()}
    if "orig_vec" in cols:
        return "new"
    if "vec" in cols:
        return "old"
    return "unknown"


def apply_schema(cur) -> None:
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    for stmt in _statements_from(sql):
        cur.execute(stmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="DROP the table even if it's already in the new schema "
                         "(wipes all vectors — use only when you mean it)")
    args = ap.parse_args()

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            state = detect_state(cur)
            print(f"Detected schema state: {state}")

            if args.force and state != "missing":
                print("  --force: DROPing existing table")
                cur.execute("DROP TABLE stock_image_vectors")
                state = "missing"

            if state == "new":
                print("  Schema already current — nothing to do.")
                return

            if state == "old":
                print("  Old 512-d schema present. Vectors are incompatible "
                      "with the new model; dropping.")
                cur.execute("DROP TABLE stock_image_vectors")

            if state == "unknown":
                print("  Table exists but neither old nor new shape detected.")
                print("  Refusing to touch it. Pass --force to wipe.")
                return

            print("  Creating new schema...")
            apply_schema(cur)
        conn.commit()

    print("Done.")


if __name__ == "__main__":
    main()
