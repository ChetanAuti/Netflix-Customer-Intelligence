"""Customer-level feature table, built at ONE shared snapshot date for every customer.

Trailing windows before the snapshot: viewing 90d, payments 180d, tickets/feedback 365d.
No outcome columns (subscription_status, cancellation_*, churn_reason, churn_date) are used.

A shared snapshot is deliberate: an earlier version used each churned customer's own churn_date
as the cutoff, which leaked the label through window length (AUC 0.99 - too good to be true).
src/models/leakage_check.py documents the strictly forward-looking variants (snapshots before
the churn event), where the signal is much weaker in this dataset.

  * SCORE_SNAPSHOT (end of data) -> features used for training AND scoring (current state)
  * TRAIN_SNAPSHOT (2025-10-01)  -> only used by leakage_check.py (forward-looking experiment)

Run:  python -m src.features.build_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import AS_OF_DATE, PROCESSED

AS_OF = pd.Timestamp(AS_OF_DATE)
TRAIN_SNAPSHOT = pd.Timestamp("2025-10-01")
SCORE_SNAPSHOT = AS_OF + pd.Timedelta(days=1)  # include sessions on the last day

VIEW_DAYS, PAY_DAYS, TICKET_DAYS = 90, 180, 365


def load() -> dict[str, pd.DataFrame]:
    dates = {
        "customers": ["registration_date"],
        "subscriptions": ["subscription_start_date", "subscription_end_date", "cancellation_date"],
        "viewing_activity": ["viewing_date"], "payments": ["payment_date"],
        "support_tickets": ["ticket_date"], "customer_feedback": ["feedback_date"],
        "churn_labels": ["churn_date"],
    }
    names = ["customers", "subscription_plans", "subscriptions", "content", "viewing_activity",
             "payments", "support_tickets", "customer_feedback", "churn_labels"]
    return {n: pd.read_csv(PROCESSED / f"{n}.csv", parse_dates=dates.get(n, [])) for n in names}


def _norm(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100


def build(snapshot: pd.Timestamp, t: dict[str, pd.DataFrame] | None = None,
          drop_prior_churners: bool = True) -> pd.DataFrame:
    """One row per customer that is registered and subscribed before `snapshot`."""
    t = t or load()
    S = pd.Timestamp(snapshot)
    first_sub = t["subscriptions"].groupby("customer_id").subscription_start_date.min()
    df = t["customers"].merge(t["churn_labels"], on="customer_id", how="left")
    df = df[df.customer_id.map(first_sub) < S].copy()
    # exclude anyone who had already cancelled before the snapshot
    if drop_prior_churners:
        df = df[~(df.churned & (df.churn_date < S))]
    df["snapshot"] = S
    df["customer_tenure_days"] = (S - df.registration_date).dt.days.clip(lower=0)
    ids = df.customer_id

    # ---------------- viewing (trailing 90d) ----------------
    v = t["viewing_activity"].merge(t["content"][["content_id", "genre"]], on="content_id", how="left")
    v = v[v.customer_id.isin(ids) & (v.viewing_date < S) & (v.viewing_date >= S - pd.Timedelta(days=VIEW_DAYS))].copy()
    v["days_before"] = (S - v.viewing_date).dt.days
    v["month"] = v.viewing_date.dt.to_period("M")
    g = v.groupby("customer_id")
    f = pd.DataFrame({
        "session_count": g.size(),
        "total_watch_minutes": g.watch_duration_minutes.sum(),
        "avg_watch_time_per_session": g.watch_duration_minutes.mean(),
        "avg_session_minutes": g.session_duration_minutes.mean(),
        "avg_completion_pct": g.completion_percentage.mean(),
        "distinct_titles": g.content_id.nunique(),
        "genre_diversity": g.genre.nunique(),
        "device_diversity": g.device_type.nunique(),
        "min_days_before": g.days_before.min(),
        "binge_share": g.watch_duration_minutes.apply(lambda s: (s >= 120).mean()),
    })
    for name, lo, hi in [("w1", 0, 30), ("w2", 30, 60), ("w3", 60, 90)]:
        m = v[(v.days_before >= lo) & (v.days_before < hi)]
        f[f"watch_{name}"] = m.groupby("customer_id").watch_duration_minutes.sum()
        f[f"sessions_{name}"] = m.groupby("customer_id").size()
    df = df.merge(f, left_on="customer_id", right_index=True, how="left")
    zero_cols = ["session_count", "total_watch_minutes", "avg_watch_time_per_session", "avg_session_minutes",
                 "distinct_titles", "genre_diversity", "device_diversity", "binge_share", "watch_w1", "watch_w2",
                 "watch_w3", "sessions_w1", "sessions_w2", "sessions_w3"]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["avg_completion_pct"] = df.avg_completion_pct.fillna(df.avg_completion_pct.median())
    df["no_viewing_data"] = (df.session_count == 0).astype(int)
    df["days_since_last_activity"] = df.min_days_before.fillna(VIEW_DAYS)  # no sessions in 90d -> capped at 90
    df["login_frequency"] = df.session_count / (VIEW_DAYS / 30)             # sessions per 30 days
    df["monthly_watch_time_trend"] = (df.watch_w1 - df.watch_w3) / 2        # slope in minutes per 30d bucket
    df["watch_trend_pct"] = (df.watch_w1 - df.watch_w3) / (df.watch_w3 + df.watch_w1 + 1)
    df["watch_last30_vs_prev60"] = df.watch_w1 / ((df.watch_w2 + df.watch_w3) / 2 + 1)
    df = df.drop(columns=["min_days_before"])

    # ---------------- payments (trailing 180d) ----------------
    p = t["payments"][t["payments"].customer_id.isin(ids)]
    p = p[(p.payment_date < S) & (p.payment_date >= S - pd.Timedelta(days=PAY_DAYS))].copy()
    p["failed"] = (p.payment_status == "Failed").astype(int)
    p["days_before"] = (S - p.payment_date).dt.days
    gp = p.groupby("customer_id")
    pf = pd.DataFrame({
        "payment_count": gp.size(), "failed_payment_count": gp.failed.sum(), "failed_payment_rate": gp.failed.mean(),
        "failed_last_30d": p[p.days_before < 30].groupby("customer_id").failed.sum(),
        "distinct_payment_methods": gp.payment_method.nunique(),
    })
    df = df.merge(pf, left_on="customer_id", right_index=True, how="left")
    pay_cols = ["payment_count", "failed_payment_count", "failed_payment_rate", "failed_last_30d", "distinct_payment_methods"]
    df[pay_cols] = df[pay_cols].fillna(0)
    # lifetime value to date: all successful payments in the dataset (target of the CLV model)
    ok = t["payments"][t["payments"].payment_status == "Success"].groupby("customer_id").amount.sum()
    df["customer_lifetime_value"] = df.customer_id.map(ok).fillna(0.0)
    # spend up to the snapshot (usable as an input for scoring, never the target)
    ok_pre = t["payments"][(t["payments"].payment_status == "Success") & (t["payments"].payment_date < S)]
    df["spend_to_snapshot"] = df.customer_id.map(ok_pre.groupby("customer_id").amount.sum()).fillna(0.0)

    # ---------------- support (trailing 365d) ----------------
    s = t["support_tickets"][t["support_tickets"].customer_id.isin(ids)]
    s = s[(s.ticket_date < S) & (s.ticket_date >= S - pd.Timedelta(days=TICKET_DAYS))].copy()
    s["billing"] = (s.issue_category == "Billing & Payments").astype(int)
    s["unresolved"] = s.ticket_status.isin(["Pending", "Escalated"]).astype(int)
    s["csat"] = s.customer_satisfaction_score.astype(float)
    gs = s.groupby("customer_id")
    sf = pd.DataFrame({
        "support_ticket_count": gs.size(), "avg_customer_satisfaction": gs.csat.mean(),
        "avg_resolution_hours": gs.resolution_time_hours.mean(), "unresolved_ticket_count": gs.unresolved.sum(),
        "billing_ticket_share": gs.billing.mean(), "low_csat_ticket_count": gs.csat.apply(lambda x: (x <= 2).sum()),
    })
    df = df.merge(sf, left_on="customer_id", right_index=True, how="left")
    sup_cols = ["support_ticket_count", "unresolved_ticket_count", "billing_ticket_share", "low_csat_ticket_count"]
    df[sup_cols] = df[sup_cols].fillna(0)
    df["csat_missing"] = df.avg_customer_satisfaction.isna().astype(int)
    df["avg_customer_satisfaction"] = df.avg_customer_satisfaction.fillna(3.0)
    df["avg_resolution_hours"] = df.avg_resolution_hours.fillna(0.0)

    # ---------------- feedback (trailing 365d) ----------------
    fb = t["customer_feedback"][t["customer_feedback"].customer_id.isin(ids)]
    fb = fb[(fb.feedback_date < S) & (fb.feedback_date >= S - pd.Timedelta(days=TICKET_DAYS))]
    gf = fb.groupby("customer_id")
    df = df.merge(pd.DataFrame({"feedback_count": gf.size(), "avg_feedback_rating": gf.rating.mean()}),
                  left_on="customer_id", right_index=True, how="left")
    df["feedback_count"] = df.feedback_count.fillna(0)
    df["avg_feedback_rating"] = df.avg_feedback_rating.fillna(3.0)

    # ---------------- subscription state at the snapshot ----------------
    sub = t["subscriptions"].merge(t["subscription_plans"][["plan_id", "plan_name"]], on="plan_id")
    sub = sub[sub.customer_id.isin(ids) & (sub.subscription_start_date < S)].sort_values(["customer_id", "subscription_start_date"])
    last = sub.groupby("customer_id").tail(1).set_index("customer_id")
    first = sub.groupby("customer_id").head(1).set_index("customer_id")
    df["plan_name"] = df.customer_id.map(last.plan_name)
    df["monthly_price"] = df.customer_id.map(last.monthly_price)
    df["auto_renew"] = df.customer_id.map(last.auto_renew).astype(float)
    df["plan_changes"] = df.customer_id.map(sub.groupby("customer_id").size() - 1).fillna(0)
    rank = {"Basic": 1, "Standard": 2, "Premium": 3}
    df["plan_direction"] = (df.plan_name.map(rank) - df.customer_id.map(first.plan_name).map(rank)).fillna(0)

    # ---------------- composite engagement score (0-100) ----------------
    recency = 100 - _norm(df.days_since_last_activity)
    df["engagement_score"] = (0.4 * recency + 0.3 * _norm(df.login_frequency)
                              + 0.3 * _norm(df.avg_watch_time_per_session * df.avg_completion_pct)).round(2)
    df["engagement_tier"] = pd.cut(df.engagement_score, [-1, 33, 66, 101], labels=["Low", "Medium", "High"]).astype(str)
    df["tenure_bucket"] = pd.cut(df.customer_tenure_days, [-1, 180, 365, 730, 1460, 10_000],
                                 labels=["<6m", "6-12m", "1-2y", "2-4y", "4y+"]).astype(str)
    df["churned"] = df.churned.astype(int)
    return df.reset_index(drop=True)


NUMERIC = [
    "age", "customer_tenure_days", "session_count", "total_watch_minutes", "avg_watch_time_per_session",
    "avg_session_minutes", "avg_completion_pct", "distinct_titles", "genre_diversity", "device_diversity",
    "binge_share", "days_since_last_activity", "login_frequency", "monthly_watch_time_trend", "watch_trend_pct",
    "watch_last30_vs_prev60", "watch_w1", "watch_w2", "watch_w3", "sessions_w1", "sessions_w2", "sessions_w3",
    "payment_count", "failed_payment_count", "failed_payment_rate", "failed_last_30d", "distinct_payment_methods",
    "support_ticket_count", "avg_customer_satisfaction", "avg_resolution_hours", "unresolved_ticket_count",
    "billing_ticket_share", "low_csat_ticket_count", "csat_missing", "feedback_count", "avg_feedback_rating",
    "monthly_price", "auto_renew", "plan_changes", "plan_direction", "engagement_score", "no_viewing_data",
]
CATEGORICAL = ["plan_name", "country", "acquisition_channel", "customer_segment", "gender"]


def run() -> pd.DataFrame:
    """Current-state feature table for ALL 8,000 customers (snapshot = end of data)."""
    df = build(SCORE_SNAPSHOT, load(), drop_prior_churners=False)
    df.to_csv(PROCESSED / "customer_features.csv", index=False)
    return df


if __name__ == "__main__":
    f = run()
    print(f.shape, "churn rate", f.churned.mean().round(4), "| NaNs in model inputs:", int(f[NUMERIC].isna().sum().sum()))
