"""Capability 5 - root-cause analysis agent ("why did churn increase in <country> last month?").

Fixed investigation plan (each step is a SQL query whose result is stored as evidence):
  1 churn delta  ->  2 pre-churn engagement  ->  3 payment failures  ->  4 support experience
  ->  5 plan-mix & reason-mix shifts  ->  6 explanation.
Guardrail: every number in the final text must appear in the evidence; otherwise the LLM text is rejected
and a deterministic, evidence-only explanation is returned instead.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

import pandas as pd

from src.database.connection import query_df
from src.llm.client import get_llm
from src.llm.reports import default_end_date

ACTIVE_AT = "c.registration_date < '{ms}' AND (cl.churned = FALSE OR cl.churn_date >= '{ms}')"
CHURNED_IN = "cl.churned = TRUE AND cl.churn_date >= '{ms}' AND cl.churn_date < '{me}'"


def _bounds(month: str):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(month).strip()):
        raise ValueError("month must look like YYYY-MM, e.g. 2026-05")
    y, m = map(int, month.strip().split("-"))
    ms = date(y, m, 1)
    me = date(y + (m == 12), m % 12 + 1, 1)
    ps = date(y - (m == 1), (m - 2) % 12 + 1, 1)
    return ps, ms, me


def countries() -> list[str]:
    return query_df("SELECT DISTINCT country FROM customers ORDER BY 1").country.tolist()


def _where(country: str | None) -> str:
    if country is None:
        return ""
    if country not in countries():          # whitelist -> no SQL injection through the region name
        raise ValueError(f"unknown country: {country}")
    return f"AND c.country = '{country}'"


def _churn_rate(ms: date, me: date, w: str) -> dict:
    r = query_df(f"""SELECT COUNT(*) base, SUM(CASE WHEN {CHURNED_IN.format(ms=ms, me=me)} THEN 1 ELSE 0 END) churned
                     FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id
                     WHERE {ACTIVE_AT.format(ms=ms)} {w}""").iloc[0]
    base, ch = int(r.base), int(r.churned or 0)
    return {"active_at_start": base, "churned": ch, "churn_rate_pct": round(100 * ch / base, 2) if base else None}


def pick_target(end: date | None = None) -> tuple[str, str]:
    """Month = latest complete month with billing data; country = largest churn-rate increase (>=150 customers)."""
    end = end or default_end_date()
    last = end if (end + timedelta(days=1)).day == 1 else end.replace(day=1) - timedelta(days=1)
    month = f"{last.year}-{last.month:02d}"
    best, best_delta = None, -1e9
    for c in countries():
        n = int(query_df(f"SELECT COUNT(*) n FROM customers c WHERE c.country = '{c}'").n.iloc[0])
        if n < 150:
            continue
        ps, ms, me = _bounds(month)
        cur, prev = _churn_rate(ms, me, _where(c)), _churn_rate(ps, ms, _where(c))
        if cur["churn_rate_pct"] is None or prev["churn_rate_pct"] is None:
            continue
        d = cur["churn_rate_pct"] - prev["churn_rate_pct"]
        if d > best_delta:
            best, best_delta = c, d
    return best, month


def _group_compare(ms: date, me: date, lo: date, hi: date, w: str, table: str, col: str, fn: str) -> pd.DataFrame:
    return query_df(f"""
        SELECT g.grp, COUNT(*) n, AVG(COALESCE(x.v, 0)) avg_value
        FROM (SELECT c.customer_id,
                     CASE WHEN {CHURNED_IN.format(ms=ms, me=me)} THEN 'churned_in_month' ELSE 'retained' END grp
              FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id
              WHERE {ACTIVE_AT.format(ms=ms)} {w}) g
        LEFT JOIN (SELECT customer_id, {fn} v FROM {table} WHERE {col} >= '{lo}' AND {col} < '{hi}' GROUP BY customer_id) x
               ON x.customer_id = g.customer_id
        GROUP BY g.grp""").set_index("grp")


def investigate(country: str | None, month: str) -> dict:
    ps, ms, me = _bounds(month)
    w = _where(country)
    ev: dict = {"scope": country or "All countries", "month": month, "previous_month": f"{ps.year}-{ps.month:02d}", "steps": {}}

    # 1. churn delta
    cur, prev = _churn_rate(ms, me, w), _churn_rate(ps, ms, w)
    ev["steps"]["1_churn"] = {"current": cur, "previous": prev}

    # 2. engagement of the people who left vs those who stayed (viewing in the 30 days before the month)
    eng = _group_compare(ms, me, ms - timedelta(days=30), ms, w, "viewing_activity", "viewing_date", "SUM(watch_duration_minutes)")
    if "churned_in_month" in eng.index and "retained" in eng.index:
        a, b = float(eng.loc["churned_in_month", "avg_value"]), float(eng.loc["retained", "avg_value"])
        ev["steps"]["2_engagement"] = {"churners_avg_watch_min_prior_30d": round(a, 1), "retained_avg_watch_min_prior_30d": round(b, 1),
                                       "gap_pct": round(100 * (a - b) / b, 1) if b else None}
    # base-level engagement trend: minutes per active customer, month vs previous month
    def per_cust(lo, hi):
        r = query_df(f"""SELECT COALESCE(SUM(v.watch_duration_minutes), 0) m FROM viewing_activity v JOIN customers c ON c.customer_id = v.customer_id
                         WHERE v.viewing_date >= '{lo}' AND v.viewing_date < '{hi}' {w}""").iloc[0].m
        return float(r)
    base_cur, base_prev = cur["active_at_start"], prev["active_at_start"]
    if base_cur and base_prev:
        m1, m0 = per_cust(ms, me) / base_cur, per_cust(ps, ms) / base_prev
        ev["steps"]["2_engagement"] = ev["steps"].get("2_engagement", {}) | {
            "watch_min_per_customer_current": round(m1, 1), "watch_min_per_customer_previous": round(m0, 1),
            "watch_change_pct": round(100 * (m1 - m0) / m0, 1) if m0 else None}

    # 3. payments (failed share in the 60 days before the month)
    pay = _group_compare(ms, me, ms - timedelta(days=60), ms, w, "payments", "payment_date",
                         "SUM(CASE WHEN payment_status = 'Failed' THEN 1.0 ELSE 0 END)")
    if "churned_in_month" in pay.index and "retained" in pay.index:
        ev["steps"]["3_payments"] = {"churners_failed_payments_per_customer_prior_60d": round(float(pay.loc["churned_in_month", "avg_value"]), 3),
                                     "retained_failed_payments_per_customer_prior_60d": round(float(pay.loc["retained", "avg_value"]), 3)}
    pf = query_df(f"""SELECT AVG(CASE WHEN p.payment_status = 'Failed' THEN 100.0 ELSE 0 END) rate, COUNT(*) n FROM payments p
                      JOIN customers c ON c.customer_id = p.customer_id WHERE p.payment_date >= '{ms}' AND p.payment_date < '{me}' {w}""").iloc[0]
    pp = query_df(f"""SELECT AVG(CASE WHEN p.payment_status = 'Failed' THEN 100.0 ELSE 0 END) rate FROM payments p
                      JOIN customers c ON c.customer_id = p.customer_id WHERE p.payment_date >= '{ps}' AND p.payment_date < '{ms}' {w}""").iloc[0]
    if pf.n:
        ev["steps"].setdefault("3_payments", {}).update({"failure_rate_pct_current": round(float(pf.rate), 2),
                                                          "failure_rate_pct_previous": None if pd.isna(pp.rate) else round(float(pp.rate), 2)})
    else:
        ev["steps"]["3_payments"] = {"note": "no billing records for this month"}

    # 4. support experience
    def tk(lo, hi):
        r = query_df(f"""SELECT COUNT(*) n, AVG(t.customer_satisfaction_score) csat FROM support_tickets t JOIN customers c ON c.customer_id = t.customer_id
                         WHERE t.ticket_date >= '{lo}' AND t.ticket_date < '{hi}' {w}""").iloc[0]
        return int(r.n), (None if pd.isna(r.csat) else round(float(r.csat), 2))
    (n1, c1), (n0, c0) = tk(ms, me), tk(ps, ms)
    ev["steps"]["4_support"] = {"tickets_current": n1, "tickets_previous": n0, "avg_csat_current": c1, "avg_csat_previous": c0}

    # 5. plan mix and churn reasons among cancellations
    def mix(lo, hi, expr, extra=""):
        return query_df(f"""SELECT {expr} k, COUNT(*) n FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id {extra}
                            WHERE cl.churned = TRUE AND cl.churn_date >= '{lo}' AND cl.churn_date < '{hi}' {w} GROUP BY 1 ORDER BY n DESC""")
    plan_join = ("JOIN (SELECT s.customer_id, sp.plan_name FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id "
                 "JOIN (SELECT customer_id, MAX(subscription_start_date) d FROM subscriptions GROUP BY customer_id) l "
                 "ON l.customer_id = s.customer_id AND l.d = s.subscription_start_date) pl ON pl.customer_id = c.customer_id")
    for name, expr, extra in [("plan_mix", "pl.plan_name", plan_join), ("reason_mix", "cl.churn_reason", "")]:
        a, b = mix(ms, me, expr, extra), mix(ps, ms, expr, extra)
        ta, tb = max(int(a.n.sum()), 1), max(int(b.n.sum()), 1)
        shares = {}
        for k in set(a.k) | set(b.k):
            sa = 100 * int(a.n[a.k == k].sum()) / ta
            sb = 100 * int(b.n[b.k == k].sum()) / tb
            shares[k] = {"share_pct_current": round(sa, 1), "share_pct_previous": round(sb, 1), "shift_pp": round(sa - sb, 1)}
        ev["steps"][f"5_{name}"] = dict(sorted(shares.items(), key=lambda kv: -abs(kv[1]["shift_pp"]))[:4])
    ev["findings"] = _findings(ev)
    return ev


def _findings(ev: dict) -> list[str]:
    s, out = ev["steps"], []
    c, p = s["1_churn"]["current"], s["1_churn"]["previous"]
    if c["churn_rate_pct"] is not None and p["churn_rate_pct"] is not None:
        d = round(c["churn_rate_pct"] - p["churn_rate_pct"], 2)
        out.append(f"Churn moved from {p['churn_rate_pct']}% to {c['churn_rate_pct']}% ({d:+} pts); {c['churned']} cancellations out of {c['active_at_start']} customers at the start of the month.")
        if c["churned"] < 30:
            out.append(f"Small sample: only {c['churned']} cancellations, so month-to-month moves can be noise.")
    e = s.get("2_engagement", {})
    if e.get("gap_pct") is not None and e["gap_pct"] < -10:
        out.append(f"Customers who cancelled watched {abs(e['gap_pct'])}% less than retained customers in the 30 days before the month ({e['churners_avg_watch_min_prior_30d']} vs {e['retained_avg_watch_min_prior_30d']} minutes).")
    if e.get("watch_change_pct") is not None and e["watch_change_pct"] < -10:
        out.append(f"Watch time per customer fell {abs(e['watch_change_pct'])}% month over month.")
    pay = s.get("3_payments", {})
    a, b = pay.get("churners_failed_payments_per_customer_prior_60d"), pay.get("retained_failed_payments_per_customer_prior_60d")
    if a is not None and b is not None and b > 0 and a / b > 1.3:
        out.append(f"Cancelling customers had more failed payments before leaving ({a} vs {b} per customer in the prior 60 days).")
    sup = s["4_support"]
    if sup["avg_csat_current"] is not None and sup["avg_csat_previous"] is not None and sup["avg_csat_current"] - sup["avg_csat_previous"] <= -0.2:
        out.append(f"Support satisfaction dropped from {sup['avg_csat_previous']} to {sup['avg_csat_current']}.")
    for name, label in [("5_plan_mix", "plan"), ("5_reason_mix", "reason")]:
        for k, v in list(s[name].items())[:1]:
            if abs(v["shift_pp"]) >= 5:
                out.append(f"Among cancellations, the share from {label} '{k}' moved {v['shift_pp']:+} pts ({v['share_pct_previous']}% -> {v['share_pct_current']}%).")
    if len(out) == 1:
        out.append("No other signal crossed its alert threshold, so the change cannot be attributed to a specific driver from this data.")
    return out


# ------------------------------------------------------------------ number guard
def _numbers(obj) -> set[str]:
    nums: set[str] = set()
    def add(x: float):
        for d in (0, 1, 2, 3):
            nums.add(f"{abs(x):.{d}f}")
        nums.add(str(abs(x)))
        nums.add(f"{abs(x):,.0f}")
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            add(float(o))
        elif isinstance(o, str):
            for m in re.findall(r"\d+(?:\.\d+)?", o):
                nums.add(m)
    walk(obj)
    return nums


def unverified_numbers(text: str, evidence: dict) -> list[str]:
    allowed = _numbers(evidence)
    cleaned = re.sub(r"\d{4}-\d{2}(-\d{2})?", " ", text)           # months/dates we supplied ourselves
    found = [n.rstrip(".").replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", cleaned)]
    ok = {a.replace(",", "") for a in allowed}
    return [n for n in found if n not in ok]


def explain(ev: dict) -> dict:
    deterministic = f"Root-cause analysis for {ev['scope']}, {ev['month']}:\n" + "\n".join(f"- {f}" for f in ev["findings"])
    llm = get_llm()
    if not llm.available:
        return {"text": deterministic, "source": "rules", "unverified_numbers": []}
    prompt = ("Explain why churn changed, as a short analyst note (max 140 words). Use ONLY numbers from this evidence JSON; "
              "say when a signal is weak or the sample is small. Evidence:\n" + json.dumps(ev, default=str))
    text = llm.complete(prompt, system="You are a careful analyst. Never state a number that is not in the evidence.", max_tokens=450)
    bad = unverified_numbers(text, ev)
    if bad:
        return {"text": deterministic, "source": "rules (LLM text rejected: unverified numbers " + ", ".join(bad[:5]) + ")", "unverified_numbers": bad}
    return {"text": text, "source": "llm", "unverified_numbers": []}


def run(country: str | None = None, month: str | None = None) -> dict:
    if country is None or month is None:
        c, m = pick_target()
        country, month = country or c, month or m
    ev = investigate(country, month)
    return {"evidence": ev, **explain(ev)}
