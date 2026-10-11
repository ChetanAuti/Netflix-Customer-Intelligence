import json

import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL, NUMERIC
from src.utils.config import PROCESSED, REPORTS

FORBIDDEN = {"churned", "churn_date", "churn_reason", "subscription_status", "cancellation_date", "cancellation_reason",
             "customer_lifetime_value", "spend_to_snapshot"}


def test_feature_table_shape_and_no_missing_inputs():
    f = pd.read_csv(PROCESSED / "customer_features.csv")
    assert len(f) == 8000 and f.customer_id.is_unique
    assert f[NUMERIC + CATEGORICAL].notna().all().all()


def test_no_outcome_columns_used_as_inputs():
    assert not FORBIDDEN & set(NUMERIC + CATEGORICAL)


def test_clv_inputs_exclude_payment_amounts():
    from src.models.common import CLV_CAT, CLV_NUM
    assert not FORBIDDEN & set(CLV_NUM + CLV_CAT)
    assert "payment_count" not in CLV_NUM


def test_scores_are_valid_probabilities():
    s = pd.read_csv(PROCESSED / "customer_scores.csv")
    assert len(s) == 8000 and s.churn_probability.between(0, 1).all()
    assert set(s.risk_tier) == {"High", "Medium", "Low"}
    assert abs((s.risk_tier == "High").mean() - 0.15) < 0.01


def test_model_beats_random_and_is_not_suspiciously_perfect():
    m = json.loads((REPORTS / "model_metrics.json").read_text())["churn"]
    auc = m["test_metrics_at_f1_threshold"]["roc_auc"]
    assert 0.55 < auc < 0.9, "AUC outside plausible range - check for leakage"


def test_segments_cover_everyone():
    s = pd.read_csv(PROCESSED / "customer_scores.csv")
    assert s.segment.notna().all() and s.segment.nunique() >= 4
