"""Tiny DB access layer. PostgreSQL in production, SQLite for a zero-setup local demo.

DATABASE_URL examples
  postgresql://user:password@localhost:5432/netflix
  sqlite:///data/netflix.db            (default when DATABASE_URL is not set)
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from urllib.parse import urlparse

import pandas as pd

from src.utils.config import ROOT

DEFAULT_URL = f"sqlite:///{ROOT / 'data' / 'netflix.db'}"


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_URL)


def dialect() -> str:
    return "postgres" if database_url().startswith(("postgres://", "postgresql")) else "sqlite"


def connect():
    url = database_url()
    if dialect() == "sqlite":
        path = url.replace("sqlite:///", "", 1)
        return sqlite3.connect(path, check_same_thread=False)
    import psycopg2  # imported lazily so SQLite-only setups do not need it
    return psycopg2.connect(url)


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def query_df(sql: str, params: tuple | dict | None = None, timeout_s: int = 15) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame (with a statement timeout on PostgreSQL)."""
    import time
    with session() as conn:
        if dialect() == "postgres":
            cur = conn.cursor()
            cur.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
            cur.execute("SET default_transaction_read_only = on")
            cur.close()
        else:  # SQLite has no statement_timeout: abort via a progress handler, and open the DB read-only
            deadline = time.time() + timeout_s
            conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20_000)
            conn.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(sql, conn, params=params)


# --- small dialect helpers so the same analytics code runs on PostgreSQL and SQLite --------------
def month_of(col: str) -> str:
    return f"to_char({col}, 'YYYY-MM')" if dialect() == "postgres" else f"strftime('%Y-%m', {col})"


def quarter_of(col: str) -> str:
    if dialect() == "postgres":
        return f"to_char({col}, 'YYYY\"Q\"Q')"
    return f"strftime('%Y', {col}) || 'Q' || ((CAST(strftime('%m', {col}) AS INTEGER) + 2) / 3)"
