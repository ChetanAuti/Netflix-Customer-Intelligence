"""Load the cleaned CSVs (+ ML scores) into PostgreSQL (or a local SQLite demo DB).

  python database/load_data.py --target postgres      # needs DATABASE_URL and psycopg2
  python database/load_data.py --target sqlite        # creates data/netflix.db (no server needed)

Run `python -m src.data_processing.clean`, `python -m src.features.build_features` and
`python -m src.models.train` first (the last one writes customer_scores.csv).
"""
from __future__ import annotations

import argparse
import io
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import PROCESSED  # noqa: E402

LOAD_ORDER = ["customers", "subscription_plans", "subscriptions", "content", "viewing_activity", "payments",
              "support_tickets", "customer_feedback", "churn_labels", "customer_scores"]
INT_COLS = {
    "customers": ["age"], "subscription_plans": ["max_devices"], "content": ["release_year", "duration_minutes"],
    "viewing_activity": ["watch_duration_minutes", "session_duration_minutes"],
    "support_tickets": ["customer_satisfaction_score"], "customer_feedback": ["rating"],
}
DATE_COLS = {"customers": ["registration_date"], "subscriptions": ["subscription_start_date", "subscription_end_date", "cancellation_date"],
             "viewing_activity": ["viewing_date"], "payments": ["payment_date"], "support_tickets": ["ticket_date"],
             "customer_feedback": ["feedback_date"], "churn_labels": ["churn_date"]}


def schema_columns() -> dict[str, list[str]]:
    """Column order per table, parsed from schema.sql."""
    ddl = (ROOT / "database" / "schema.sql").read_text()
    cols = {}
    for m in re.finditer(r"CREATE TABLE (\w+) \((.*?)\n\);", ddl, flags=re.S):
        names = []
        for line in m.group(2).splitlines():
            for part in re.split(r",\s+(?=[a-z_]+ [A-Z])", line.strip()):
                tok = part.strip().split()
                if tok and tok[0].islower() and not tok[0].startswith(("constraint", "--")) and len(tok) > 1 and tok[1][0].isupper():
                    names.append(tok[0])
        cols[m.group(1)] = names
    return cols


def read_table(name: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    if name == "customer_scores":
        df = df[["customer_id", "churn_probability", "churn_risk_score", "risk_tier", "segment", "predicted_clv"]]
    df = df[cols].copy()
    for c in INT_COLS.get(name, []):
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    return df


def load_postgres(url: str):
    import psycopg2
    cols = schema_columns()
    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute((ROOT / "database" / "schema.sql").read_text())
    for name in LOAD_ORDER:
        df = read_table(name, cols[name])
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        cur.copy_expert(f"COPY {name} ({', '.join(cols[name])}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)
        print(f"  {name:20s} {len(df):>7,} rows")
    conn.commit()
    cur.execute("ANALYZE")
    conn.commit()
    conn.close()


def load_sqlite(path: Path):
    cols = schema_columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    for name in LOAD_ORDER:
        df = read_table(name, cols[name])
        df.to_sql(name, conn, index=False)
        print(f"  {name:20s} {len(df):>7,} rows")
    for tbl, col in [("subscriptions", "customer_id"), ("viewing_activity", "customer_id"), ("payments", "customer_id"),
                     ("support_tickets", "customer_id"), ("customer_feedback", "customer_id"), ("viewing_activity", "viewing_date"),
                     ("payments", "payment_date")]:
        conn.execute(f"CREATE INDEX idx_{tbl}_{col} ON {tbl} ({col})")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["postgres", "sqlite"], default="sqlite")
    ap.add_argument("--url", default=None, help="PostgreSQL URL (defaults to $DATABASE_URL)")
    a = ap.parse_args()
    if a.target == "postgres":
        import os
        url = a.url or os.environ["DATABASE_URL"]
        print("Loading into PostgreSQL ..."); load_postgres(url)
    else:
        print("Loading into SQLite demo DB ..."); load_sqlite(ROOT / "data" / "netflix.db")
    print("done")
