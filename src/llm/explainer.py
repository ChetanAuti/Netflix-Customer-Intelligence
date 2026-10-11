"""Capability 3 - ML prediction explainer.

churn model output (probability + drivers) -> business-language explanation.
Drivers are computed by single-feature substitution: replace one input by the population median/mode and
measure how much the predicted risk moves (a lightweight, model-agnostic alternative to SHAP).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.llm.client import get_llm
from src.models.common import CHURN_CAT, CHURN_NUM
from src.utils.config import MODELS, PROCESSED

FRIENDLY = {
    "monthly_watch_time_trend": "falling watch time", "watch_trend_pct": "falling watch time", "watch_last30_vs_prev60": "recent watch time vs earlier",
    "watch_w1": "watch time in the last 30 days", "watch_w2": "watch time 30-60 days ago", "watch_w3": "watch time 60-90 days ago",
    "days_since_last_activity": "days since last viewing", "login_frequency": "how often they watch", "session_count": "number of sessions",
    "total_watch_minutes": "total watch time", "engagement_score": "overall engagement", "failed_payment_rate": "failed-payment rate",
    "failed_payment_count": "failed payments", "failed_last_30d": "failed payments in the last 30 days",
    "avg_customer_satisfaction": "support satisfaction (CSAT)", "support_ticket_count": "support ticket volume",
    "low_csat_ticket_count": "unhappy support outcomes", "customer_tenure_days": "tenure", "auto_renew": "auto-renew setting",
    "monthly_price": "plan price", "plan_name": "plan", "avg_completion_pct": "completion of what they start",
    "avg_watch_time_per_session": "session length", "unresolved_ticket_count": "unresolved tickets",
}

_cache: dict = {}


def _load():
    if not _cache:
        _cache["churn"] = joblib.load(MODELS / "churn_model.joblib")
        _cache["feats"] = pd.read_csv(PROCESSED / "customer_features.csv").set_index("customer_id", drop=False)
        _cache["scores"] = pd.read_csv(PROCESSED / "customer_scores.csv").set_index("customer_id")
    return _cache


def _prob(m, X: pd.DataFrame) -> np.ndarray:
    raw = np.clip(m["pipeline"].predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return m["calibrator"].predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]


def customer_facts(customer_id: str) -> dict:
    """Plain facts about the customer (every number the explanation may use)."""
    c = _load()
    r = c["feats"].loc[customer_id]
    sc = c["scores"].loc[customer_id]
    w1, w3 = float(r.watch_w1), float(r.watch_w3)
    facts = {
        "customer_id": customer_id, "plan": r.plan_name, "country": r.country, "tenure_days": int(r.customer_tenure_days),
        "auto_renew": bool(r.auto_renew), "segment": sc.segment, "risk_tier": sc.risk_tier,
        "churn_probability_pct": round(float(sc.churn_probability) * 100, 1),
        "watch_minutes_last_30d": int(w1), "watch_minutes_60_90d_ago": int(w3),
        "watch_change_pct": None if w3 == 0 else round((w1 - w3) / w3 * 100),
        "days_since_last_activity": int(r.days_since_last_activity), "sessions_last_90d": int(r.session_count),
        "failed_payments_last_180d": int(r.failed_payment_count), "failed_payments_last_30d": int(r.failed_last_30d),
        "support_tickets_last_year": int(r.support_ticket_count),
        "avg_csat": None if r.csat_missing else round(float(r.avg_customer_satisfaction), 1),
        "predicted_clv_usd": round(float(sc.predicted_clv), 2),
    }
    return facts


def top_drivers(customer_id: str, n: int = 5) -> list[dict]:
    c = _load()
    m = c["churn"]
    feats = c["feats"]
    row = feats.loc[[customer_id], CHURN_NUM + CHURN_CAT]
    base = float(_prob(m, row)[0])
    ref = {col: feats[col].median() for col in CHURN_NUM} | {col: feats[col].mode().iloc[0] for col in CHURN_CAT}
    out = []
    for col in CHURN_NUM + CHURN_CAT:
        alt = row.copy()
        alt[col] = ref[col]
        delta = base - float(_prob(m, alt)[0])
        val, typ = row[col].iloc[0], ref[col]
        out.append({"feature": col, "value": val.item() if hasattr(val, "item") else val,
                    "typical": typ.item() if hasattr(typ, "item") else typ, "risk_effect_pts": round(delta * 100, 2)})
    out.sort(key=lambda d: -abs(d["risk_effect_pts"]))
    return out[:n]


def explain_customer(customer_id: str) -> dict:
    facts = customer_facts(customer_id)
    drivers = top_drivers(customer_id)
    up = [d for d in drivers if d["risk_effect_pts"] > 0]
    down = [d for d in drivers if d["risk_effect_pts"] < 0]
    llm = get_llm()
    if llm.available:
        prompt = (f"Explain this customer's churn risk to a retention manager in 4-6 sentences.\nFACTS (the only numbers you may use): {facts}\n"
                  f"MODEL DRIVERS (positive = raises risk; points of probability): {[(FRIENDLY.get(d['feature'], d['feature']), d['risk_effect_pts']) for d in drivers]}\n"
                  "Name the biggest risk-raising and protective factors in plain language. Do not invent numbers.")
        text = llm.complete(prompt, system="You explain ML predictions to non-technical managers, faithfully.", max_tokens=450)
    else:
        text = _template_explanation(facts, up, down)
    return {"facts": facts, "drivers": drivers, "explanation": text}


def _template_explanation(f: dict, up: list[dict], down: list[dict]) -> str:
    lvl = {"High": "high", "Medium": "moderate", "Low": "low"}[f["risk_tier"]]
    s = [f"Customer {f['customer_id']} ({f['plan']} plan, {f['country']}) has a {lvl} churn risk of about {f['churn_probability_pct']}%."]
    bits = []
    if f["watch_change_pct"] is not None and f["watch_change_pct"] <= -20:
        bits.append(f"viewing fell {abs(f['watch_change_pct'])}% ({f['watch_minutes_60_90d_ago']} -> {f['watch_minutes_last_30d']} minutes per 30 days)")
    if f["days_since_last_activity"] >= 30:
        bits.append(f"nothing watched for {f['days_since_last_activity']} days")
    if f["failed_payments_last_180d"] > 0:
        bits.append(f"{f['failed_payments_last_180d']} failed payment(s) in the last 180 days")
    if f["avg_csat"] is not None and f["avg_csat"] <= 2.5:
        bits.append(f"low support satisfaction ({f['avg_csat']}/5)")
    if not f["auto_renew"]:
        bits.append("auto-renew is off")
    if bits:
        s.append("Warning signs: " + "; ".join(bits) + ".")
    if up:
        s.append("The model's biggest risk-raising factors are " + ", ".join(FRIENDLY.get(d["feature"], d["feature"]) for d in up[:3]) + ".")
    if down:
        s.append("Protective factors: " + ", ".join(FRIENDLY.get(d["feature"], d["feature"]) for d in down[:2]) + ".")
    s.append(f"Segment: {f['segment']}; predicted lifetime value ${f['predicted_clv_usd']:.0f}.")
    return " ".join(s)
