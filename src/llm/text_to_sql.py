"""Capability 1 - Text-to-SQL with a non-negotiable validation layer.

question -> LLM writes SQL against the known schema -> validate (SELECT only, blocked verbs, schema check)
-> execute with timeout + row cap -> LLM turns the result into a plain-language answer.
Every attempt is appended to reports/logs/query_audit.jsonl (question, SQL, status).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from src.database import connection
from src.llm.client import get_llm
from src.utils.config import LOG_DIR

MAX_ROWS = 500
TIMEOUT_S = 15

SCHEMA: dict[str, list[str]] = {
    "customers": ["customer_id", "first_name", "last_name", "age", "gender", "country", "state_or_region", "city",
                  "registration_date", "acquisition_channel", "customer_segment", "preferred_language"],
    "subscription_plans": ["plan_id", "plan_name", "monthly_price", "video_quality", "max_devices", "advertisements"],
    "subscriptions": ["subscription_id", "customer_id", "plan_id", "subscription_start_date", "subscription_end_date",
                      "subscription_status", "auto_renew", "cancellation_date", "cancellation_reason", "monthly_price"],
    "content": ["content_id", "title", "content_type", "genre", "release_year", "duration_minutes", "content_language",
                "production_country", "maturity_rating"],
    "viewing_activity": ["viewing_id", "customer_id", "content_id", "viewing_date", "watch_duration_minutes",
                         "completion_percentage", "device_type", "login_location", "session_duration_minutes"],
    "payments": ["payment_id", "customer_id", "subscription_id", "payment_date", "amount", "payment_method",
                 "payment_status", "failed_payment_reason"],
    "support_tickets": ["ticket_id", "customer_id", "ticket_date", "issue_category", "issue_subcategory", "priority",
                        "ticket_status", "resolution_time_hours", "customer_satisfaction_score", "ticket_description",
                        "support_agent_id"],
    "customer_feedback": ["feedback_id", "customer_id", "feedback_date", "rating", "feedback_text", "sentiment_label"],
    "churn_labels": ["customer_id", "churned", "churn_date", "churn_reason"],
    "customer_scores": ["customer_id", "churn_probability", "churn_risk_score", "risk_tier", "segment", "predicted_clv"],
}

BLOCKED = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "copy", "merge",
           "replace", "attach", "detach", "pragma", "vacuum", "call", "execute", "into"]

SQL_KEYWORDS = set("""select from where group by order having limit offset join left right inner outer full cross on using as and or not
in is null like ilike between case when then else end distinct union all intersect except with exists asc desc nulls first last
over partition rows range unbounded preceding following current row filter within true false interval date timestamp cast
extract year month day week quarter hour epoch to_char date_trunc coalesce nullif round count sum avg min max lag lead rank
dense_rank row_number ntile greatest least abs floor ceil length lower upper substr substring strftime julianday date_part now
current_date integer int numeric float text varchar boolean bigint decimal real double precision lateral recursive window current any some values natural""".split())


@dataclass
class SQLResult:
    question: str
    sql: str = ""
    status: str = "ok"           # ok | blocked | error | no_llm
    error: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    answer: str = ""
    attempts: int = 1


def schema_prompt() -> str:
    return "\n".join(f"{t}({', '.join(c)})" for t, c in SCHEMA.items())


# ----------------------------------------------------------------------------- validation
class SQLValidationError(ValueError):
    pass


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql: str, schema: dict[str, list[str]] = SCHEMA) -> str:
    """Return the cleaned SQL or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("empty query")
    s = sql.strip().rstrip(";").strip()
    s = re.sub(r"--[^\n]*", "", s)            # drop line comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    if ";" in _strip_literals(s):
        raise SQLValidationError("multiple statements are not allowed")
    bare = _strip_literals(s).lower()
    if not re.match(r"^\s*\(?\s*(select|with)\b", bare):
        raise SQLValidationError("only SELECT statements are allowed")
    for kw in BLOCKED:
        if re.search(rf"\b{kw}\b", bare):
            raise SQLValidationError(f"forbidden keyword: {kw.upper()}")

    # ---- schema check: tables
    ctes = set(re.findall(r"\b(\w+)\s+as\s*\(", bare))
    scan = re.sub(r"\b(extract|substring|trim|overlay)\s*\([^()]*\)", " ", bare)  # EXTRACT(YEAR FROM col) is not a table
    tables = re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", scan)
    for t in tables:
        if t not in schema and t not in ctes:
            raise SQLValidationError(f"unknown table: {t}")
    # ---- schema check: alias.column references
    aliases: dict[str, str] = {}
    for t, a in re.findall(r"(?=\b(?:from|join)\s+([a-z_][a-z0-9_]*)(?:\s+as)?\s+([a-z_][a-z0-9_]*))", scan):
        if a not in SQL_KEYWORDS:
            aliases[a] = t
    for t in tables:
        aliases.setdefault(t, t)
    out_aliases = set(re.findall(r"\bas\s+([a-z_][a-z0-9_]*)", bare)) | ctes
    for a, c in re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", bare):
        t = aliases.get(a)
        if t in schema and c not in schema[t] and c != "*":
            raise SQLValidationError(f"unknown column: {a}.{c} (table {t})")
    # ---- schema check: every other identifier must be a known column / alias / keyword / function
    known_cols = {c for cols in schema.values() for c in cols}
    funcs = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", bare))
    # identifiers inside date/interval/extract style expressions are covered by SQL_KEYWORDS
    for tok in set(re.findall(r"\b[a-z_][a-z0-9_]*\b", bare)):
        if tok in known_cols or tok in schema or tok in SQL_KEYWORDS or tok in aliases or tok in out_aliases or tok in funcs:
            continue
        if re.search(rf"\b\d+\s+{tok}s?\b", bare) or tok.rstrip("s") in SQL_KEYWORDS:   # e.g. interval '3 months'
            continue
        raise SQLValidationError(f"unknown identifier: {tok}")
    return s


def add_row_cap(sql: str, max_rows: int = MAX_ROWS) -> str:
    return f"SELECT * FROM (\n{sql}\n) AS _capped LIMIT {max_rows}"


# ----------------------------------------------------------------------------- audit log
def audit(question: str, sql: str, status: str, error: str = "", rows: int | None = None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question, "sql": sql, "status": status,
           "error": error, "rows": rows}
    with open(LOG_DIR / "query_audit.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------- generation
SYSTEM = f"""You translate business questions into ONE read-only SQL SELECT for a streaming-subscription database.
Rules: output only the SQL in a ```sql block. Use only these tables/columns:
{schema_prompt()}
Notes: churn_labels.churned is boolean; dates are DATE (ISO); the data ends on 2026-06-30, so never use CURRENT_DATE -
anchor "recent" periods on MAX(date) of the relevant table. payment_status is 'Success' or 'Failed'.
customer_scores.risk_tier is 'High'/'Medium'/'Low'. Add LIMIT for long lists. No comments, no DDL/DML."""


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()


# tiny offline library so the app is demo-able without an API key
OFFLINE_TEMPLATES = [
    (("churn", "country"), "SELECT c.country, COUNT(*) AS customers, SUM(CASE WHEN cl.churned THEN 1 ELSE 0 END) AS churned, "
     "ROUND(100.0 * SUM(CASE WHEN cl.churned THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct FROM customers c "
     "JOIN churn_labels cl ON cl.customer_id = c.customer_id GROUP BY c.country ORDER BY churn_rate_pct DESC LIMIT 10"),
    (("churn", "plan"), "SELECT sp.plan_name, COUNT(*) AS customers, ROUND(100.0 * SUM(CASE WHEN cl.churned THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct "
     "FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id JOIN churn_labels cl ON cl.customer_id = s.customer_id "
     "WHERE s.subscription_end_date IS NULL OR s.subscription_status <> 'Plan Changed' GROUP BY sp.plan_name ORDER BY churn_rate_pct DESC"),
    (("churn", "reason"), "SELECT churn_reason, COUNT(*) AS customers FROM churn_labels WHERE churned GROUP BY churn_reason ORDER BY customers DESC"),
    (("revenue", "plan"), "SELECT sp.plan_name, COUNT(*) AS active_subscriptions, SUM(s.monthly_price) AS monthly_revenue FROM subscriptions s "
     "JOIN subscription_plans sp ON sp.plan_id = s.plan_id WHERE s.subscription_status = 'Active' GROUP BY sp.plan_name ORDER BY monthly_revenue DESC"),
    (("most", "watched"), "SELECT c.title, SUM(v.watch_duration_minutes) AS total_minutes FROM viewing_activity v JOIN content c ON c.content_id = v.content_id "
     "GROUP BY c.title ORDER BY total_minutes DESC LIMIT 10"),
    (("failed", "payment"), "SELECT failed_payment_reason, COUNT(*) AS failed_payments FROM payments WHERE payment_status = 'Failed' GROUP BY failed_payment_reason ORDER BY failed_payments DESC"),
    (("high", "risk"), "SELECT segment, COUNT(*) AS high_risk_customers FROM customer_scores WHERE risk_tier = 'High' GROUP BY segment ORDER BY high_risk_customers DESC"),
    (("complaint",), "SELECT issue_subcategory, COUNT(*) AS tickets, ROUND(AVG(customer_satisfaction_score), 2) AS avg_csat FROM support_tickets GROUP BY issue_subcategory ORDER BY tickets DESC LIMIT 10"),
    (("device",), "SELECT device_type, SUM(watch_duration_minutes) AS total_minutes FROM viewing_activity GROUP BY device_type ORDER BY total_minutes DESC"),
    (("active", "customers"), "SELECT c.country, COUNT(DISTINCT c.customer_id) AS active_customers FROM customers c JOIN subscriptions s ON s.customer_id = c.customer_id "
     "WHERE s.subscription_status = 'Active' GROUP BY c.country ORDER BY active_customers DESC LIMIT 15"),
]


def generate_sql(question: str, feedback: str = "") -> str:
    llm = get_llm()
    if llm.available:
        prompt = f"Question: {question}" + (f"\n\nYour previous SQL was rejected: {feedback}\nFix it." if feedback else "")
        return extract_sql(llm.complete(prompt, system=SYSTEM, max_tokens=700))
    q = question.lower()
    for keys, sql in OFFLINE_TEMPLATES:
        if all(k in q for k in keys):
            return sql
    return ""


def summarize(question: str, sql: str, df: pd.DataFrame) -> str:
    llm = get_llm()
    if df.empty:
        return "The query ran but returned no rows."
    if llm.available:
        prompt = (f"Question: {question}\nSQL used: {sql}\nResult (first 25 rows, CSV):\n{df.head(25).to_csv(index=False)}\n"
                  f"Total rows: {len(df)}. Answer the question in 2-4 plain-English sentences. Use only numbers present in the result.")
        return llm.complete(prompt, system="You are a careful business analyst. Never invent numbers.", max_tokens=350)
    top = df.iloc[0]
    parts = ", ".join(f"{k} = {v}" for k, v in top.items())
    return f"The query returned {len(df)} row(s). Top row: {parts}."


def ask(question: str, execute: bool = True) -> SQLResult:
    res = SQLResult(question=question)
    feedback = ""
    for attempt in (1, 2):
        res.attempts = attempt
        try:
            raw = generate_sql(question, feedback)
        except Exception as exc:  # network / API errors
            res.status, res.error = "error", f"LLM error: {exc}"
            audit(question, "", res.status, res.error)
            return res
        if not raw:
            res.status, res.error = "no_llm", "No API key configured and no offline template matches this question."
            audit(question, "", res.status, res.error)
            return res
        res.sql = raw
        try:
            clean = validate_sql(raw)
        except SQLValidationError as exc:
            res.status, res.error = "blocked", str(exc)
            audit(question, raw, "blocked", str(exc))
            feedback = str(exc)
            continue
        try:
            res.df = connection.query_df(add_row_cap(clean), timeout_s=TIMEOUT_S) if execute else pd.DataFrame()
        except Exception as exc:
            res.status, res.error = "error", str(exc)
            audit(question, clean, "error", str(exc))
            feedback = str(exc)
            continue
        res.sql, res.status, res.error = clean, "ok", ""
        audit(question, clean, "ok", rows=len(res.df))
        res.answer = summarize(question, clean, res.df)
        return res
    return res
