"""Exploratory data analysis: generates 24 figures into reports/figures/eda_*.png.

Run:  python -m src.utils.eda
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.build_features import load
from src.utils.config import FIGURES, PROCESSED

RED, GREY = "#E50914", "#564d4d"
sns.set_theme(style="whitegrid", rc={"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
_n = 0


def save(title: str, name: str):
    global _n
    _n += 1
    plt.title(title, fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / f"eda_{_n:02d}_{name}.png")
    plt.close()


def bar(series: pd.Series, title: str, name: str, horizontal=False, color=RED, fmt=None, size=(7, 4)):
    plt.figure(figsize=size)
    (series.plot.barh if horizontal else series.plot.bar)(color=color)
    if horizontal:
        plt.gca().invert_yaxis()
    else:
        plt.xticks(rotation=40, ha="right")
    plt.xlabel(""); plt.ylabel("")
    if fmt:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt.format(v)))
    save(title, name)


def run() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in FIGURES.glob("eda_*.png"):
        f.unlink()
    t = load()
    c, s, p, v, tk, fb, ch, plans, content = (t["customers"], t["subscriptions"], t["payments"], t["viewing_activity"],
                                              t["support_tickets"], t["customer_feedback"], t["churn_labels"],
                                              t["subscription_plans"], t["content"])
    feats = pd.read_csv(PROCESSED / "customer_features.csv")
    cc = c.merge(ch, on="customer_id")
    active = s[s.subscription_status == "Active"].merge(plans[["plan_id", "plan_name"]], on="plan_id")

    bar(c.country.value_counts().head(15), "Customers by country (top 15)", "customers_by_country", horizontal=True)
    plt.figure(figsize=(7, 4)); plt.hist(c.age, bins=30, color=RED); plt.xlabel("Age"); save("Customer age distribution", "age")
    bar(c.acquisition_channel.value_counts(), "Customers by acquisition channel", "acquisition_channel")
    bar(c.customer_segment.value_counts(), "Customers by account type", "customer_type", color=GREY)
    bar(active.plan_name.value_counts(), "Active subscriptions by plan", "plan_distribution", color=GREY)
    rev = active.groupby("plan_name").monthly_price.sum()
    plt.figure(figsize=(5.5, 4.5)); plt.pie(rev, labels=rev.index, autopct="%1.0f%%", colors=[GREY, "#b20710", RED]); save("Monthly revenue mix by plan", "revenue_mix")

    can = ch[ch.churned].groupby(ch.churn_date.dt.to_period("M")).size()
    plt.figure(figsize=(8, 4)); can.plot(marker="o", color=RED); plt.xlabel(""); save("Cancellations per month", "cancellation_trend")
    plan_churn = cc.merge(feats[["customer_id", "plan_name"]], on="customer_id").groupby("plan_name").churned.mean()
    bar(plan_churn, "Churn rate by plan", "churn_by_plan", color=RED, fmt="{:.0%}")

    plt.figure(figsize=(7, 4)); plt.hist(v.watch_duration_minutes, bins=50, color=RED); plt.xlabel("Minutes watched per session"); save("Watch-time distribution (after outlier capping)", "watch_time")
    vc = v.merge(content[["content_id", "content_type", "genre"]], on="content_id")
    bar(vc.groupby("content_type").completion_percentage.mean().sort_values(), "Average completion % by content type", "completion_by_type", color=GREY)
    bar(v.groupby("device_type").watch_duration_minutes.sum().sort_values(ascending=False) / 1000, "Total watch time by device (thousand min)", "device_usage")
    plt.figure(figsize=(8, 4)); v.groupby(v.viewing_date.dt.to_period("M")).watch_duration_minutes.sum().div(1000).plot(marker="o", color=RED)
    plt.ylabel("thousand min"); plt.xlabel(""); save("Monthly watch time", "watch_trend")

    ctry = cc.groupby("country").churned.agg(["mean", "size"]).query("size>=100")["mean"].sort_values(ascending=False).head(12)
    bar(ctry, "Churn rate by country (n>=100)", "churn_by_country", horizontal=True, fmt="{:.0%}")
    order = ["<6m", "6-12m", "1-2y", "2-4y", "4y+"]
    bar(feats.groupby("tenure_bucket").churned.mean().reindex(order), "Churn rate by tenure bucket", "churn_by_tenure", fmt="{:.0%}")
    bar(feats.groupby("engagement_tier").churned.mean().reindex(["Low", "Medium", "High"]), "Churn rate by engagement tier", "churn_by_engagement", fmt="{:.0%}")
    fails = p[p.payment_status == "Failed"].groupby("customer_id").size()
    cc["fail_bucket"] = pd.cut(cc.customer_id.map(fails).fillna(0), [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    bar(cc.groupby("fail_bucket", observed=True).churned.mean(), "Churn rate by failed-payment history", "churn_by_failed_payments", fmt="{:.0%}")
    cc["tk_bucket"] = pd.cut(cc.customer_id.map(tk.groupby("customer_id").size()).fillna(0), [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    bar(cc.groupby("tk_bucket", observed=True).churned.mean(), "Churn rate by support-ticket volume", "churn_by_tickets", fmt="{:.0%}")

    plt.figure(figsize=(8, 4)); tk.dropna(subset=["ticket_date"]).groupby(tk.ticket_date.dt.to_period("Q")).size().plot(marker="o", color=RED)
    plt.xlabel(""); save("Support tickets per quarter", "ticket_trend")
    bar(tk.groupby("issue_category").resolution_time_hours.mean().sort_values(), "Average resolution time (hours) by category", "resolution_by_category", horizontal=True, color=GREY)
    g = tk.groupby("issue_category").agg(n=("ticket_id", "size"), csat=("customer_satisfaction_score", "mean")).sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(g.index, g.n, color=RED); ax.tick_params(axis="x", rotation=30)
    ax2 = ax.twinx(); ax2.plot(g.index, g.csat, "o-", color="black"); ax2.set_ylim(2, 4); ax2.set_ylabel("Avg CSAT"); ax2.grid(False)
    save("Complaint categories: volume (bars) vs CSAT (line)", "complaints_vs_csat")
    bar(ch[ch.churned].churn_reason.value_counts(), "Churn reasons", "churn_reasons", horizontal=True)
    plt.figure(figsize=(6, 4)); sns.countplot(x="rating", hue="sentiment_label", data=fb, palette=[RED, GREY, "#999"]); save("Feedback ratings by (corrected) sentiment", "feedback_ratings")
    fm = p[p.payment_status == "Failed"].groupby(p.payment_date.dt.to_period("Q")).size()
    plt.figure(figsize=(8, 4)); fm.plot(marker="o", color=RED); plt.xlabel(""); save("Failed payments per quarter", "failed_payments")

    cols = ["customer_tenure_days", "engagement_score", "login_frequency", "avg_watch_time_per_session", "watch_trend_pct",
            "failed_payment_rate", "support_ticket_count", "avg_customer_satisfaction", "monthly_price", "customer_lifetime_value", "churned"]
    plt.figure(figsize=(8, 6.5)); sns.heatmap(feats[cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    save("Correlation of key customer features with churn", "correlation_heatmap")
    return _n


if __name__ == "__main__":
    print(run(), "figures written to", FIGURES)
