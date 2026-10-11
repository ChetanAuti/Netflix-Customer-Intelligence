"""Data cleaning pipeline: data/raw/*.csv -> data/processed/*.csv.

Every decision is recorded in reports/logs/cleaning_log.json so the strategy for
each column is documented and reproducible. Run:  python -m src.data_processing.clean
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.config import LOG_DIR, PROCESSED, RAW

COUNTRY_MAP = {
    "united states": "United States", "usa": "United States", "us": "United States", "u.s.a": "United States",
    "india": "India", "in": "India", "bharat": "India",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
    "canada": "Canada", "ca": "Canada", "brazil": "Brazil", "br": "Brazil",
    "germany": "Germany", "de": "Germany", "france": "France", "fr": "France",
    "mexico": "Mexico", "mx": "Mexico", "japan": "Japan", "jp": "Japan",
    "philippines": "Philippines", "ph": "Philippines", "indonesia": "Indonesia", "id": "Indonesia",
    "italy": "Italy", "it": "Italy", "nigeria": "Nigeria", "ng": "Nigeria", "spain": "Spain", "es": "Spain",
    "south africa": "South Africa", "za": "South Africa", "australia": "Australia", "au": "Australia",
    "south korea": "South Korea", "s. korea": "South Korea", "kr": "South Korea",
    "argentina": "Argentina", "ar": "Argentina", "poland": "Poland", "pl": "Poland",
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates", "u.a.e": "United Arab Emirates",
}

DEVICE_MAP = {
    "smart tv": "Smart TV", "smarttv": "Smart TV", "tv": "Smart TV",
    "mobile": "Mobile", "smartphone": "Mobile", "phone": "Mobile",
    "laptop": "Laptop", "desktop": "Laptop", "pc": "Laptop",
    "tablet": "Tablet", "ipad": "Tablet",
    "gaming console": "Gaming Console", "console": "Gaming Console", "playstation/xbox": "Gaming Console",
}


def canon_country(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().map(COUNTRY_MAP)


def _mode(x: pd.Series):
    m = x.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def clean_customers(cust: pd.DataFrame, viewing: pd.DataFrame, log: dict) -> pd.DataFrame:
    c = cust.copy()
    log["customers"] = {"raw_rows": len(c)}
    c = c.drop_duplicates()  # the 48 duplicates are exact copies
    assert c.customer_id.is_unique
    log["customers"]["duplicates_removed"] = len(cust) - len(c)

    # country: canonicalise, then impute from the customer's most common login country
    raw_country = c.country
    c["country"] = canon_country(raw_country)
    unmapped = raw_country.notna() & c.country.isna()
    assert not unmapped.any(), raw_country[unmapped].unique()
    login = viewing.assign(loc=canon_country(viewing.login_location)).groupby("customer_id")["loc"].agg(_mode)
    missing = c.country.isna()
    c.loc[missing, "country"] = c.loc[missing, "customer_id"].map(login)
    log["customers"]["country_missing_raw"] = int(raw_country.isna().sum())
    log["customers"]["country_imputed_from_login_location"] = int(missing.sum() - c.country.isna().sum())
    c["country"] = c.country.fillna("Unknown")
    log["customers"]["country_still_unknown"] = int((c.country == "Unknown").sum())

    # age: implausible values -> NaN, then median by customer_segment
    bad_age = (c.age < 13) | (c.age > 100)
    log["customers"]["age_outliers_nulled"] = int(bad_age.sum())
    c.loc[bad_age, "age"] = np.nan
    c["age_imputed"] = c.age.isna()
    log["customers"]["age_imputed"] = int(c.age_imputed.sum())
    c["age"] = c.age.fillna(c.groupby("customer_segment").age.transform("median")).round().astype("Int64")

    # city: unknown label; language: mode within country, else English
    log["customers"]["city_missing_filled_unknown"] = int(c.city.isna().sum())
    c["city"] = c.city.fillna("Unknown")
    lang_by_country = c.groupby("country").preferred_language.agg(_mode)
    miss_lang = c.preferred_language.isna()
    log["customers"]["language_missing_filled_country_mode"] = int(miss_lang.sum())
    c.loc[miss_lang, "preferred_language"] = c.loc[miss_lang, "country"].map(lang_by_country)
    c["preferred_language"] = c.preferred_language.fillna("English")
    c["state_or_region"] = np.nan  # shipped blank for all rows
    c["registration_date"] = pd.to_datetime(c.registration_date)
    return c


def clean_viewing(view: pd.DataFrame, log: dict) -> pd.DataFrame:
    v = view.copy()
    log["viewing_activity"] = {"raw_rows": len(v)}
    raw_dev = v.device_type
    v["device_type"] = raw_dev.str.strip().str.lower().map(DEVICE_MAP)
    assert not (raw_dev.notna() & v.device_type.isna()).any()
    log["viewing_activity"]["device_inconsistent_fixed"] = int((raw_dev.notna() & (raw_dev != v.device_type)).sum())
    n_missing = int(v.device_type.isna().sum())
    cust_mode = v.groupby("customer_id").device_type.agg(_mode)
    v["device_type"] = v.device_type.fillna(v.customer_id.map(cust_mode)).fillna("Unknown")
    log["viewing_activity"]["device_missing"] = n_missing
    log["viewing_activity"]["device_still_unknown"] = int((v.device_type == "Unknown").sum())

    v["login_location"] = canon_country(v.login_location).fillna("Unknown")

    # completion_percentage: median of the title, else global median (+ flag)
    v["completion_imputed"] = v.completion_percentage.isna()
    log["viewing_activity"]["completion_imputed"] = int(v.completion_imputed.sum())
    med_title = v.groupby("content_id").completion_percentage.transform("median")
    v["completion_percentage"] = v.completion_percentage.fillna(med_title).fillna(v.completion_percentage.median())

    # watch / session outliers: cap at the IQR upper fence (Q3 + 3*IQR)
    for col in ["watch_duration_minutes", "session_duration_minutes"]:
        q1, q3 = v[col].quantile([0.25, 0.75])
        fence = q3 + 3 * (q3 - q1)
        out = v[col] > fence
        log["viewing_activity"][f"{col}_outliers_capped"] = int(out.sum())
        log["viewing_activity"][f"{col}_cap_value"] = float(fence)
        v[f"{col}_was_outlier"] = out
        v[col] = v[col].clip(upper=fence)
    # a session cannot be shorter than the time actually watched
    fix = v.session_duration_minutes < v.watch_duration_minutes
    log["viewing_activity"]["session_lt_watch_fixed"] = int(fix.sum())
    v.loc[fix, "session_duration_minutes"] = v.loc[fix, "watch_duration_minutes"]
    v["viewing_date"] = pd.to_datetime(v.viewing_date)
    return v


def resolve_payments(pay: pd.DataFrame, subs: pd.DataFrame, log: dict) -> pd.DataFrame:
    p = pay.drop(columns=["subscription_id"]).copy()
    log["payments"] = {"raw_rows": len(p)}
    p["payment_date"] = pd.to_datetime(p.payment_date)
    s = subs[["subscription_id", "customer_id", "subscription_start_date", "monthly_price"]].copy()
    s["subscription_start_date"] = pd.to_datetime(s.subscription_start_date)
    s = s.sort_values("subscription_start_date")
    p = p.sort_values("payment_date")
    # subscription with the latest start_date <= payment_date (the active window)
    m = pd.merge_asof(p, s, left_on="payment_date", right_on="subscription_start_date",
                      by="customer_id", direction="backward")
    # payments dated before the first segment start -> attach to the customer's first subscription
    first = s.drop_duplicates("customer_id").set_index("customer_id")
    miss = m.subscription_id.isna()
    log["payments"]["before_first_subscription_start"] = int(miss.sum())
    m.loc[miss, "subscription_id"] = m.loc[miss, "customer_id"].map(first.subscription_id)
    m.loc[miss, "monthly_price"] = m.loc[miss, "customer_id"].map(first.monthly_price)
    log["payments"]["subscription_id_resolved"] = int(m.subscription_id.notna().sum())

    # anomalous amounts (0, proration, refunds, entry errors): any amount that is not one of the
    # three catalogue prices and is far from the segment price is replaced by the segment price
    ratio = m.amount / m.monthly_price
    common = m.amount.round(2).isin([6.99, 15.49, 22.99])
    anomaly = ~common & ((m.amount == 0) | (ratio > 1.5) | (ratio < 0.9) | (ratio != 1))
    log["payments"]["amount_anomalies_replaced_with_segment_price"] = int(anomaly.sum())
    m["amount_anomaly"] = anomaly
    m.loc[anomaly, "amount"] = m.loc[anomaly, "monthly_price"]
    m = m.drop(columns=["subscription_start_date", "monthly_price"])
    return m.sort_values("payment_id").reset_index(drop=True)


def clean_tickets(t: pd.DataFrame, log: dict) -> pd.DataFrame:
    t = t.copy()
    log["support_tickets"] = {"raw_rows": len(t)}
    d = pd.to_datetime(t.ticket_date, errors="coerce")
    log["support_tickets"]["invalid_dates_set_null"] = int(d.isna().sum())
    t["ticket_date"] = d
    t["ticket_date_invalid"] = d.isna()
    # CSAT / resolution time exist only for Resolved / Closed tickets: those NULLs are expected
    log["support_tickets"]["csat_null_unresolved_kept_null"] = int(t.customer_satisfaction_score.isna().sum())
    t["customer_satisfaction_score"] = t.customer_satisfaction_score.astype("Int64")
    return t


def clean_feedback(f: pd.DataFrame, log: dict) -> pd.DataFrame:
    f = f.copy()
    log["customer_feedback"] = {"raw_rows": len(f)}
    f["sentiment_label_raw"] = f.sentiment_label
    derived = pd.cut(f.rating, [0, 2, 3, 5], labels=["Negative", "Neutral", "Positive"]).astype(str)
    log["customer_feedback"]["sentiment_relabelled_from_rating"] = int((derived != f.sentiment_label).sum())
    f["sentiment_label"] = derived
    f["feedback_date"] = pd.to_datetime(f.feedback_date)
    return f


def run() -> dict:
    log: dict = {}
    names = ["customers", "subscription_plans", "subscriptions", "content", "viewing_activity",
             "payments", "support_tickets", "customer_feedback", "churn_labels"]
    raw = {n: pd.read_csv(RAW / f"{n}.csv") for n in names}
    view_c = clean_viewing(raw["viewing_activity"], log)
    cust_c = clean_customers(raw["customers"], raw["viewing_activity"], log)
    subs = raw["subscriptions"].copy()
    for col in ["subscription_start_date", "subscription_end_date", "cancellation_date"]:
        subs[col] = pd.to_datetime(subs[col])
    pay_c = resolve_payments(raw["payments"], raw["subscriptions"], log)
    tick_c = clean_tickets(raw["support_tickets"], log)
    fb_c = clean_feedback(raw["customer_feedback"], log)
    churn = raw["churn_labels"].copy()
    churn["churn_date"] = pd.to_datetime(churn.churn_date)

    ids = set(cust_c.customer_id)
    log["orphan_rows_by_table"] = {
        n: int((~df.customer_id.isin(ids)).sum())
        for n, df in [("subscriptions", subs), ("viewing", view_c), ("payments", pay_c),
                      ("tickets", tick_c), ("feedback", fb_c), ("churn", churn)]}
    assert all(v == 0 for v in log["orphan_rows_by_table"].values())

    out = {"customers": cust_c, "subscription_plans": raw["subscription_plans"], "subscriptions": subs,
           "content": raw["content"], "viewing_activity": view_c, "payments": pay_c,
           "support_tickets": tick_c, "customer_feedback": fb_c, "churn_labels": churn}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for n, df in out.items():
        df.to_csv(PROCESSED / f"{n}.csv", index=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
    return log


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
