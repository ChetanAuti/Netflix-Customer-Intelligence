"""Shared helpers for model training / scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import CATEGORICAL, NUMERIC

CHURN_NUM = NUMERIC
CHURN_CAT = CATEGORICAL

# CLV model inputs: nothing derived from payment amounts or payment counts (that would leak the target)
CLV_NUM = ["age", "customer_tenure_days", "monthly_price", "auto_renew", "plan_changes", "plan_direction",
           "engagement_score", "login_frequency", "avg_watch_time_per_session", "total_watch_minutes",
           "avg_completion_pct", "watch_trend_pct", "support_ticket_count", "avg_customer_satisfaction",
           "failed_payment_rate", "feedback_count", "avg_feedback_rating", "genre_diversity", "device_diversity"]
CLV_CAT = ["plan_name", "customer_segment", "country", "acquisition_channel", "gender"]

SEGMENT_FEATURES = ["engagement_score", "login_frequency", "avg_watch_time_per_session", "customer_tenure_days",
                    "failed_payment_rate", "avg_customer_satisfaction", "monthly_price", "support_ticket_count",
                    "watch_trend_pct"]


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False), cat),
    ])


def risk_tier(prob: pd.Series, high_q: float = 0.85, med_q: float = 0.60) -> pd.Series:
    """Rank-based tiers: top 15% = High, next 25% = Medium, remainder = Low."""
    hi, md = prob.quantile(high_q), prob.quantile(med_q)
    return pd.Series(np.where(prob >= hi, "High", np.where(prob >= md, "Medium", "Low")), index=prob.index)
