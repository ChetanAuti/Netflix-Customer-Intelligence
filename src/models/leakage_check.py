"""Strictly forward-looking churn experiment (documents why the main model is a *propensity* model).

For monthly snapshots S (Oct-2025 .. Jun-2026) we build features from data BEFORE S only and label
"cancels within h days after S". Cross-validation is grouped by customer.
Result in this dataset: AUC ~0.55 - the synthetic churn label is mostly driven by behaviour measured
over the whole year, so true forward-looking signal is weak. Reported in reports/model_evaluation.md.

Run:  python -m src.models.leakage_check
"""
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

from src.features.build_features import CATEGORICAL, NUMERIC, build, load
from src.utils.config import REPORTS


def run() -> dict:
    t = load()
    panel = pd.concat([build(s, t) for s in pd.date_range("2025-10-01", "2026-06-01", freq="MS")], ignore_index=True)
    X = pd.get_dummies(panel[NUMERIC + CATEGORICAL], columns=CATEGORICAL, drop_first=True).astype(float)
    out = {}
    for h in (30, 90):
        y = (panel.churned.eq(1) & (panel.churn_date < panel.snapshot + pd.Timedelta(days=h))).astype(int)
        auc = cross_val_score(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150),
                              X, y, cv=GroupKFold(3), groups=panel.customer_id, scoring="roc_auc").mean()
        out[f"forward_{h}d"] = {"positive_rate": round(float(y.mean()), 4), "cv_roc_auc": round(float(auc), 4)}
    (REPORTS / "leakage_check.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(run())
