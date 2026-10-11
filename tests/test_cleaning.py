import numpy as np
import pandas as pd

from src.data_processing import clean


def test_country_canonicalisation():
    s = pd.Series(["USA", "united states", " India ", "UK", "uae"])
    assert clean.canon_country(s).tolist() == ["United States", "United States", "India", "United Kingdom", "United Arab Emirates"]


def test_device_map_covers_variants():
    for raw, want in {"smartphone": "Mobile", "SmartTV": "Smart TV", "Desktop": "Laptop", "ipad": "Tablet"}.items():
        assert clean.DEVICE_MAP[raw.lower()] == want


def test_feedback_sentiment_derived_from_rating():
    f = pd.DataFrame({"rating": [1, 2, 3, 4, 5], "sentiment_label": ["Positive"] * 5, "feedback_date": ["2026-01-01"] * 5})
    out = clean.clean_feedback(f, {})
    assert out.sentiment_label.tolist() == ["Negative", "Negative", "Neutral", "Positive", "Positive"]


def test_invalid_ticket_dates_become_null_and_flagged():
    t = pd.DataFrame({"ticket_date": ["2026-01-05", "31/31/2026", "not a date"], "customer_satisfaction_score": [3, None, 4]})
    out = clean.clean_tickets(t, {})
    assert out.ticket_date.isna().sum() == 2 and out.ticket_date_invalid.sum() == 2


def test_payment_resolution_and_anomaly_replacement():
    subs = pd.DataFrame({"subscription_id": ["S1", "S2"], "customer_id": ["C1", "C1"],
                         "subscription_start_date": ["2025-01-01", "2025-06-01"], "monthly_price": [6.99, 15.49]})
    pay = pd.DataFrame({"payment_id": ["P1", "P2", "P3"], "customer_id": ["C1"] * 3, "subscription_id": [None] * 3,
                        "payment_date": ["2025-02-01", "2025-07-01", "2025-08-01"], "amount": [6.99, 15.49, 170.39]})
    out = clean.resolve_payments(pay, subs, {}).set_index("payment_id")
    assert out.loc["P1", "subscription_id"] == "S1" and out.loc["P2", "subscription_id"] == "S2"
    assert out.loc["P3", "amount"] == 15.49 and bool(out.loc["P3", "amount_anomaly"])
    assert not out.loc["P1", "amount_anomaly"]


def test_processed_data_integrity():
    from src.utils.config import PROCESSED
    c = pd.read_csv(PROCESSED / "customers.csv")
    assert c.customer_id.is_unique and c.country.notna().all() and c.age.between(13, 100).all()
    v = pd.read_csv(PROCESSED / "viewing_activity.csv")
    assert (v.session_duration_minutes >= v.watch_duration_minutes).all()
    p = pd.read_csv(PROCESSED / "payments.csv")
    assert p.subscription_id.notna().all()
