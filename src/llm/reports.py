"""Capability 4 - automated business report generator.

All metrics come from SQL (calculated facts). The narrative (executive summary, findings, risks,
recommendations) is written by the LLM - or by deterministic rules when no API key is configured -
and is labelled separately from the facts.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd

from src.database.connection import query_df
from src.llm.client import get_llm
from src.utils.config import REPORTS

PERIOD_DAYS = {"weekly": 7, "monthly": 30, "quarterly": 91}


def data_end_dates() -> dict:
    q = {
        "payments": "SELECT MAX(payment_date) d FROM payments", "viewing": "SELECT MAX(viewing_date) d FROM viewing_activity",
        "tickets": "SELECT MAX(ticket_date) d FROM support_tickets", "churn": "SELECT MAX(churn_date) d FROM churn_labels",
        "feedback": "SELECT MAX(feedback_date) d FROM customer_feedback"}
    return {k: str(query_df(v).d.iloc[0])[:10] for k, v in q.items()}


def default_end_date() -> date:
    """Latest day for which billing data exists (all tables are then complete through that day)."""
    return datetime.strptime(data_end_dates()["payments"], "%Y-%m-%d").date()


def window_metrics(start: date, end_excl: date) -> dict:
    s, e = start.isoformat(), end_excl.isoformat()
    one = lambda sql: query_df(sql).iloc[0]  # noqa: E731
    active_start = int(one(f"""SELECT COUNT(*) n FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id
                               WHERE c.registration_date < '{s}' AND (cl.churned = FALSE OR cl.churn_date >= '{s}')""").n)
    cancels = int(one(f"""SELECT COUNT(*) n FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id
                          WHERE c.registration_date < '{s}' AND cl.churned = TRUE AND cl.churn_date >= '{s}' AND cl.churn_date < '{e}'""").n)
    new_c = int(one(f"SELECT COUNT(*) n FROM customers WHERE registration_date >= '{s}' AND registration_date < '{e}'").n)
    pay = one(f"""SELECT COALESCE(SUM(CASE WHEN payment_status = 'Success' THEN amount END), 0) revenue,
                         COALESCE(SUM(CASE WHEN payment_status = 'Failed' THEN 1 ELSE 0 END), 0) failed, COUNT(*) total
                  FROM payments WHERE payment_date >= '{s}' AND payment_date < '{e}'""")
    view = one(f"""SELECT COUNT(DISTINCT customer_id) viewers, COUNT(*) sessions, COALESCE(SUM(watch_duration_minutes), 0) minutes
                   FROM viewing_activity WHERE viewing_date >= '{s}' AND viewing_date < '{e}'""")
    tk = one(f"""SELECT COUNT(*) tickets, AVG(customer_satisfaction_score) csat, AVG(resolution_time_hours) res
                 FROM support_tickets WHERE ticket_date >= '{s}' AND ticket_date < '{e}'""")
    fb = one(f"""SELECT COUNT(*) n, AVG(rating) rating FROM customer_feedback WHERE feedback_date >= '{s}' AND feedback_date < '{e}'""")
    top = query_df(f"""SELECT issue_subcategory sub, COUNT(*) n FROM support_tickets WHERE ticket_date >= '{s}' AND ticket_date < '{e}'
                       GROUP BY issue_subcategory ORDER BY n DESC LIMIT 3""")
    f = lambda x, d=2: None if pd.isna(x) else round(float(x), d)  # noqa: E731
    return {
        "period": f"{s} to {(end_excl - timedelta(days=1)).isoformat()}",
        "active_customers_at_start": active_start, "new_customers": new_c, "cancellations": cancels,
        "churn_rate_pct": round(100 * cancels / active_start, 2) if active_start else None,
        "revenue_usd": f(pay.revenue), "payments": int(pay.total), "failed_payments": int(pay.failed),
        "payment_failure_rate_pct": round(100 * pay.failed / pay.total, 2) if pay.total else None,
        "active_viewers": int(view.viewers), "sessions": int(view.sessions), "watch_hours": round(float(view.minutes) / 60, 1),
        "tickets": int(tk.tickets), "avg_csat": f(tk.csat), "avg_resolution_hours": f(tk.res, 1),
        "feedback_count": int(fb.n), "avg_feedback_rating": f(fb.rating),
        "top_complaints": [{"issue": r.sub, "tickets": int(r.n)} for r in top.itertuples()],
    }


def _delta(cur, prev):
    if cur is None or prev in (None, 0):
        return None
    return round((cur - prev) / abs(prev) * 100, 1)


def generate_report(period: str = "monthly", end_date: date | None = None) -> dict:
    n = PERIOD_DAYS[period]
    end = (end_date or default_end_date())
    end_excl = end + timedelta(days=1)
    cur = window_metrics(end_excl - timedelta(days=n), end_excl)
    prev = window_metrics(end_excl - timedelta(days=2 * n), end_excl - timedelta(days=n))
    keys = ["new_customers", "cancellations", "churn_rate_pct", "revenue_usd", "payment_failure_rate_pct", "active_viewers",
            "watch_hours", "tickets", "avg_csat", "avg_feedback_rating"]
    table = pd.DataFrame([{"metric": k, "current": cur[k], "previous": prev[k], "change_pct": _delta(cur[k], prev[k])} for k in keys])
    scores = query_df("""SELECT COUNT(*) n FROM customer_scores cs JOIN churn_labels cl ON cl.customer_id = cs.customer_id
                         WHERE cs.risk_tier = 'High' AND cl.churned = FALSE""")
    facts = {"period_type": period, "current": cur, "previous": prev, "table": table.to_dict("records"),
             "high_risk_active_customers": int(scores.n.iloc[0]), "data_through": data_end_dates()}
    narrative = _narrative_llm(facts) if get_llm().available else _narrative_rules(facts)
    md = _render(period, facts, table, narrative)
    REPORTS.joinpath("generated").mkdir(parents=True, exist_ok=True)
    out = REPORTS / "generated" / f"{period}_report_{end.isoformat()}.md"
    out.write_text(md)
    return {"markdown": md, "facts": facts, "table": table, "path": str(out), "used_llm": get_llm().available}


def _narrative_llm(facts: dict) -> str:
    prompt = ("Write an executive report from these CALCULATED FACTS (JSON). Sections: **Executive summary** (3 sentences), "
              "**Key findings** (4 bullets), **Risks** (3 bullets), **Recommendations** (3 bullets). Use only numbers in the facts; "
              "if a metric is null, say data is unavailable for that period.\n" + json.dumps(facts, default=str)[:9000])
    return get_llm().complete(prompt, system="You are a precise BI analyst. Never invent a number.", max_tokens=900)


def _narrative_rules(f: dict) -> str:
    c, p = f["current"], f["previous"]
    ch = {r["metric"]: r["change_pct"] for r in f["table"]}
    def move(metric, label, good_when_up=True):
        v = ch.get(metric)
        if v is None:
            return None
        return f"{label} {'rose' if v > 0 else 'fell'} {abs(v)}% versus the previous {f['period_type']} period"
    findings = [x for x in [
        move("revenue_usd", "Revenue"), move("churn_rate_pct", "Churn rate"), move("active_viewers", "Active viewers"),
        move("payment_failure_rate_pct", "The payment failure rate"), move("tickets", "Support tickets")] if x]
    risks = []
    if (ch.get("churn_rate_pct") or 0) > 5: risks.append(f"Churn is accelerating ({c['cancellations']} cancellations, {c['churn_rate_pct']}% of the opening base).")
    if (ch.get("payment_failure_rate_pct") or 0) > 5: risks.append(f"Billing failures are up to {c['payment_failure_rate_pct']}% of attempts - involuntary churn risk.")
    if (ch.get("avg_csat") or 0) < -3: risks.append(f"Support satisfaction slipped to {c['avg_csat']}/5.")
    risks.append(f"{f['high_risk_active_customers']} active customers are currently in the High churn-risk tier.")
    top = c["top_complaints"][0]["issue"] if c["top_complaints"] else None
    recs = ["Run the payment-recovery flow for customers with recent failed payments.",
            "Target the High-risk, high-value customers with personalised outreach (see Retention page)."]
    if top: recs.append(f"Investigate the top complaint driver: '{top}'.")
    exec_sum = (f"{c['new_customers']} customers joined and {c['cancellations']} cancelled in {c['period']}, a churn rate of "
                f"{c['churn_rate_pct']}% of the opening base of {c['active_customers_at_start']:,}. Revenue was ${c['revenue_usd']:,.0f} "
                f"across {c['payments']:,} payment attempts.")
    return ("**Executive summary**\n" + exec_sum + "\n\n**Key findings**\n" + "\n".join(f"- {x}" for x in findings) +
            "\n\n**Risks**\n" + "\n".join(f"- {x}" for x in risks) + "\n\n**Recommendations**\n" + "\n".join(f"- {x}" for x in recs))


def _render(period: str, facts: dict, table: pd.DataFrame, narrative: str) -> str:
    c = facts["current"]
    fmt = lambda v: "n/a" if v is None or pd.isna(v) else (int(v) if float(v).is_integer() else v)  # noqa: E731
    rows = "\n".join(f"| {r.metric} | {fmt(r.current)} | {fmt(r.previous)} | {'' if r.change_pct is None else f'{r.change_pct:+}%'} |" for r in table.itertuples())
    top = "\n".join(f"- {t['issue']}: {t['tickets']} tickets" for t in c["top_complaints"]) or "- none"
    return (f"# {period.capitalize()} Business Report - {c['period']}\n\n"
            f"## Calculated facts (SQL)\n| metric | current | previous | change |\n|---|---|---|---|\n{rows}\n\n"
            f"Top complaints this period:\n{top}\n\nHigh-risk active customers (churn model): {facts['high_risk_active_customers']}\n\n"
            f"_Data available through: {json.dumps(facts['data_through'])}_\n\n"
            f"## AI interpretation\n_Written by {'the LLM' if get_llm().available else 'rule-based templates (no API key configured)'} "
            f"from the facts above; it is interpretation, not measured data._\n\n{narrative}\n")
