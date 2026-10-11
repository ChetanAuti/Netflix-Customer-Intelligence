"""Capability 6 - retention strategy: transparent rules first, LLM wording second."""
from __future__ import annotations

import pandas as pd

from src.llm.client import get_llm
from src.llm.explainer import customer_facts
from src.utils.config import PROCESSED


def _value_threshold() -> float:
    scores = pd.read_csv(PROCESSED / "customer_scores.csv")
    return float(scores.predicted_clv.quantile(0.75))


def recommend(customer_id: str) -> dict:
    f = customer_facts(customer_id)
    high_value = f["predicted_clv_usd"] >= _value_threshold()
    tier = f["risk_tier"]
    actions: list[dict] = []
    if tier == "High" and high_value:
        actions.append({"priority": 1, "action": "Immediate personalised outreach (account manager call + tailored offer)",
                        "why": "High churn risk on a high-value customer - the offer cost is justified."})
    elif tier == "High":
        actions.append({"priority": 1, "action": "Automated promotional campaign (e.g. 1-month discount email)",
                        "why": "High risk but lower value - use a low-cost, scalable intervention."})
    elif tier == "Medium" and high_value:
        actions.append({"priority": 2, "action": "Proactive engagement: personalised content picks and a check-in message",
                        "why": "Moderate risk on a valuable customer - nurture before it escalates."})
    else:
        actions.append({"priority": 3, "action": "No offer - keep in standard lifecycle communication",
                        "why": "Risk is low enough that an incentive would mostly be wasted spend."})
    if f["failed_payments_last_30d"] > 0 or f["failed_payments_last_180d"] >= 2:
        actions.append({"priority": 1, "action": "Payment recovery: retry flow + 'update your payment method' reminder",
                        "why": f"{f['failed_payments_last_180d']} failed payment(s) in 180 days - involuntary churn is preventable."})
    if f["avg_csat"] is not None and f["avg_csat"] <= 2.5:
        actions.append({"priority": 2, "action": "Service recovery: senior agent follow-up on the last ticket + goodwill credit",
                        "why": f"Average support satisfaction is {f['avg_csat']}/5."})
    if f["watch_change_pct"] is not None and f["watch_change_pct"] <= -30 and tier != "Low":
        actions.append({"priority": 2, "action": "Re-engagement: 'new for you' recommendations based on past genres",
                        "why": f"Watch time is down {abs(f['watch_change_pct'])}% versus 60-90 days ago."})
    if not f["auto_renew"] and tier != "Low":
        actions.append({"priority": 3, "action": "Offer an annual-plan or auto-renew incentive", "why": "Auto-renew is switched off."})
    actions.sort(key=lambda a: a["priority"])
    llm = get_llm()
    text = ""
    if llm.available:
        text = llm.complete(f"Write a short (3-5 sentence) retention plan for a manager.\nFacts: {f}\nRule-based actions: {actions}\n"
                            "Only use numbers from the facts.", system="You are a pragmatic retention strategist.", max_tokens=350)
    return {"facts": f, "high_value": high_value, "actions": actions, "plan_text": text}
