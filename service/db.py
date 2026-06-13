"""MySQL connection helper.

Env-var driven so the same code runs against:
  - local dev (a local PHP server MySQL, host.docker.internal from inside docker)
  - prod    (the web host your_db_name)

Defaults match conn.php on the host side so dev "just works" with no
config:  user=root, password='', db=poshak, port=3306.
"""

import os

import pymysql
from pymysql.cursors import DictCursor


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None else default


def get_connection():
    """Returns a fresh PyMySQL connection. Caller is responsible for
    closing it (use `with closing(conn): ...` or try/finally)."""
    return pymysql.connect(
        host=_env("DB_HOST", "host.docker.internal"),
        port=int(_env("DB_PORT", "3306")),
        user=_env("DB_USER", "root"),
        password=_env("DB_PASS", ""),
        database=_env("DB_NAME", "poshak"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )
