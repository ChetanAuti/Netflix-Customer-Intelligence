"""Train and evaluate the three models and write customer scores.

  Model 1  churn classification   (LogReg / RandomForest / HistGradientBoosting [/ XGBoost if installed])
  Model 2  customer segmentation  (K-Means, elbow + silhouette)
  Model 3  CLV regression         (Linear / RandomForest / HistGradientBoosting)

Run:  python -m src.models.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.common import CHURN_CAT, CHURN_NUM, CLV_CAT, CLV_NUM, SEGMENT_FEATURES, make_preprocessor, risk_tier
from src.utils.config import FIGURES, MODELS, PROCESSED, REPORTS

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save_fig(name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name)
    plt.close()


# ----------------------------------------------------------------------------- churn
def churn_candidates():
    cands = {
        "LogisticRegression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                               {"clf__C": [0.02, 0.1, 0.5]}),
        "RandomForest": (RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
                         {"clf__max_depth": [6, 10], "clf__min_samples_leaf": [10, 25]}),
        "GradientBoosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, early_stopping=False),
                             {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.06], "clf__max_iter": [100, 200],
                              "clf__l2_regularization": [1.0]}),
    }
    try:  # optional
        from xgboost import XGBClassifier
        cands["XGBoost"] = (XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                                          colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=1),
                            {"clf__max_depth": [2, 3], "clf__n_estimators": [150, 250]})
    except ImportError:
        pass
    return cands


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"threshold": round(float(thr), 4), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def train_churn(df: pd.DataFrame, report: dict) -> dict:
    X, y = df[CHURN_NUM + CHURN_CAT], df.churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, (est, grid) in churn_candidates().items():
        pipe = Pipeline([("prep", make_preprocessor(CHURN_NUM, CHURN_CAT)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1).fit(Xtr, ytr)
        results[name] = {"cv_roc_auc": round(gs.best_score_, 4), "best_params": gs.best_params_}
        fitted[name] = gs.best_estimator_
        print(f"  churn {name:20s} CV AUC {gs.best_score_:.4f}  {gs.best_params_}")
    best = max(results, key=lambda k: results[k]["cv_roc_auc"])
    model = fitted[best]

    oof_tr = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(ytr, oof_tr)
    p_te = model.predict_proba(Xte)[:, 1]
    for name, m in fitted.items():
        results[name]["test_roc_auc"] = round(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), 4)
    report["churn"] = {"candidates": results, "selected": best, "train_rows": len(Xtr), "test_rows": len(Xte),
                       "test_metrics_at_f1_threshold": metrics_at(yte, p_te, thr),
                       "test_metrics_at_0.5": metrics_at(yte, p_te, 0.5)}
    # decile lift on the held-out set
    d = pd.DataFrame({"y": yte.values, "p": p_te})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
    lift = d.groupby("decile", observed=True).y.agg(["mean", "sum", "size"]).rename(columns={"mean": "churn_rate"})
    lift["lift"] = lift.churn_rate / yte.mean()
    report["churn"]["decile_lift_test"] = lift.round(3).reset_index().to_dict("records")

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, m in fitted.items():
        fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1])
        ax[0].plot(fpr, tpr, label=f"{name} ({results[name]['test_roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(title="ROC curves (held-out test)", xlabel="FPR", ylabel="TPR"); ax[0].legend()
    cm = np.array(report["churn"]["test_metrics_at_f1_threshold"]["confusion_matrix"])
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, v, ha="center", va="center", fontsize=13)
    ax[1].set(title=f"Confusion matrix - {best} @ {thr:.2f}", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"], xlabel="Predicted", ylabel="Actual")
    save_fig("ml_churn_roc_confusion.png")

    plt.figure(figsize=(7, 3.6)); plt.bar(lift.index.astype(str), lift.lift); plt.axhline(1, color="k", lw=0.8)
    plt.title("Churn lift by predicted-risk decile (test set)"); plt.xlabel("Decile (1 = highest risk)"); plt.ylabel("Lift vs base rate")
    save_fig("ml_churn_lift.png")

    pi = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    report["churn"]["permutation_importance_top15"] = imp.head(15).round(4).to_dict()
    plt.figure(figsize=(7, 5)); imp.head(15)[::-1].plot.barh(color="#E50914"); plt.title("Permutation importance (drop in test AUC)")
    save_fig("ml_churn_importance.png")

    # ---- calibrated out-of-fold probabilities for ALL customers + final model refit on all data
    oof_all = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    calib = LogisticRegression().fit(np.log(oof_all / (1 - oof_all + 1e-9) + 1e-9).reshape(-1, 1), y)
    final = Pipeline(model.steps).fit(X, y)
    report["churn"]["oof_roc_auc_all_customers"] = round(roc_auc_score(y, oof_all), 4)
    return {"pipeline": final, "calibrator": calib, "threshold": thr, "oof": oof_all, "oof_calibrated": calib.predict_proba(
        np.log(oof_all / (1 - oof_all + 1e-9) + 1e-9).reshape(-1, 1))[:, 1], "numeric": CHURN_NUM, "categorical": CHURN_CAT,
        "model_name": best}


# ----------------------------------------------------------------------------- segmentation
SEGMENT_NAMES = {}


def name_segments(prof: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Business-readable names from the centroid z-scores (each rule picks the most extreme cluster)."""
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    rules = [  # (key feature, sign, min |z|, label builder, description)
        ("failed_payment_rate", 1, 0.8, lambda r: "Billing-Troubled " + ("New Signups" if r.customer_tenure_days < -0.5 else "Subscribers"),
         "Frequent failed payments (card/billing issues) - involuntary-churn risk, often recent signups."),
        ("support_ticket_count", 1, 0.8, lambda r: "Support-Heavy " + ("Satisfied" if r.avg_customer_satisfaction > 0.3 else "Frustrated") + " Users",
         "Raise many tickets; satisfaction with the outcome decides whether they stay."),
        ("engagement_score", -1, 0.8, lambda r: "At-Risk Low-Engagement",
         "Very few sessions and short watch time - quiet accounts drifting away."),
        ("engagement_score", 1, 0.5, lambda r: "High-Value Loyal",
         "Highly engaged, rising watch time, long tenure - the core of the revenue base."),
        ("watch_trend_pct", -1, 0.3, lambda r: "Fading Binge Viewers",
         "Long sessions but watch time is declining - engaged historically, cooling now."),
    ]
    names: dict[int, tuple[str, str]] = {}
    for feat, sign, thr, label, desc in rules:
        left = [k for k in z.index if int(k) not in names]
        if not left:
            break
        k = max(left, key=lambda k: sign * z.loc[k, feat])
        if sign * z.loc[k, feat] >= thr:
            names[int(k)] = (label(z.loc[k]), desc)
    for k in z.index:
        if int(k) not in names:
            names[int(k)] = ("Steady Core" if "Steady Core" not in [v[0] for v in names.values()] else f"Mixed Segment {k}",
                             "Close to average on most dimensions - the stable middle of the base.")
    return names


def train_segmentation(df: pd.DataFrame, report: dict) -> dict:
    X = df[SEGMENT_FEATURES].copy()
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    ks = list(range(2, 10)); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_, sample_size=4000, random_state=SEED))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    ax[1].plot(ks, sil, "o-", color="#E50914"); ax[1].set(title="Silhouette score", xlabel="K", ylabel="Silhouette")
    save_fig("ml_segmentation_elbow_silhouette.png")
    # choose K in the business-readable range 4-6 with the best silhouette
    cand = [k for k in ks if 4 <= k <= 6]
    K = max(cand, key=lambda k: sil[ks.index(k)])
    km = KMeans(K, n_init=20, random_state=SEED).fit(Z)
    df = df.assign(cluster=km.labels_)
    prof = df.groupby("cluster")[SEGMENT_FEATURES].mean()
    names = name_segments(prof)
    df["segment"] = df.cluster.map(lambda c: names[int(c)][0])
    summary = df.groupby("segment").agg(customers=("customer_id", "size"), churn_rate=("churned", "mean"),
                                        avg_clv=("customer_lifetime_value", "mean"),
                                        avg_price=("monthly_price", "mean")).round(3)
    report["segmentation"] = {"k_range_tested": ks, "inertia": [round(i, 1) for i in inertia],
                              "silhouette": [round(s, 4) for s in sil], "chosen_k": K,
                              "segments": {names[int(c)][0]: {"description": names[int(c)][1],
                                                               "profile": prof.loc[c].round(2).to_dict()} for c in prof.index},
                              "summary": summary.reset_index().to_dict("records")}
    pca = PCA(2, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    plt.figure(figsize=(7, 5.2))
    for s in df.segment.unique():
        m = (df.segment == s).values
        plt.scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, label=s)
    plt.legend(markerscale=3, fontsize=8); plt.title("Customer segments (PCA projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    save_fig("ml_segmentation_pca.png")
    return {"scaler": scaler, "kmeans": km, "features": SEGMENT_FEATURES, "names": {int(k): v[0] for k, v in names.items()},
            "descriptions": {v[0]: v[1] for v in names.values()}, "labels": df.segment.values}


# ----------------------------------------------------------------------------- CLV
def train_clv(df: pd.DataFrame, report: dict) -> dict:
    X, y = df[CLV_NUM + CLV_CAT], df.customer_lifetime_value
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cands = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=250, min_samples_leaf=5, random_state=SEED, n_jobs=1),
        "GradientBoosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, random_state=SEED),
    }
    res, fitted = {}, {}
    for n, est in cands.items():
        pipe = Pipeline([("prep", make_preprocessor(CLV_NUM, CLV_CAT)), ("reg", est)]).fit(Xtr, ytr)
        pr = pipe.predict(Xte)
        res[n] = {"MAE": round(mean_absolute_error(yte, pr), 2), "RMSE": round(float(np.sqrt(mean_squared_error(yte, pr))), 2),
                  "R2": round(r2_score(yte, pr), 4)}
        fitted[n] = pipe
        print(f"  clv   {n:20s} {res[n]}")
    best = min(res, key=lambda k: res[k]["RMSE"])
    report["clv"] = {"candidates": res, "selected": best, "target_mean": round(float(y.mean()), 2),
                     "target_median": round(float(y.median()), 2)}
    pr = fitted[best].predict(Xte)
    plt.figure(figsize=(5.5, 5)); plt.scatter(yte, pr, s=6, alpha=0.4); lim = [0, max(yte.max(), pr.max())]
    plt.plot(lim, lim, "k--", lw=0.8); plt.xlabel("Actual CLV ($)"); plt.ylabel("Predicted CLV ($)"); plt.title(f"CLV: actual vs predicted ({best})")
    save_fig("ml_clv_actual_vs_pred.png")
    final = Pipeline(fitted[best].steps).fit(X, y)
    return {"pipeline": final, "numeric": CLV_NUM, "categorical": CLV_CAT, "model_name": best}


def run():
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED / "customer_features.csv", parse_dates=["registration_date", "churn_date", "snapshot"])
    report: dict = {"customers": len(df), "churn_rate": round(float(df.churned.mean()), 4)}
    print("Model 1: churn"); churn = train_churn(df, report)
    print("Model 2: segmentation"); seg = train_segmentation(df, report)
    print("Model 3: CLV"); clv = train_clv(df, report)

    joblib.dump({k: v for k, v in churn.items() if k not in ("oof", "oof_calibrated")}, MODELS / "churn_model.joblib")
    joblib.dump({k: v for k, v in seg.items() if k != "labels"}, MODELS / "segmentation_model.joblib")
    joblib.dump(clv, MODELS / "clv_model.joblib")

    # customer-level scores (out-of-fold churn probabilities -> honest for existing customers)
    scores = df[["customer_id", "churned", "plan_name", "monthly_price", "country", "customer_lifetime_value"]].copy()
    scores["churn_probability"] = churn["oof_calibrated"].round(4)
    scores["churn_risk_score"] = (churn["oof_calibrated"] * 100).round(1)
    scores["risk_tier"] = risk_tier(pd.Series(churn["oof_calibrated"], index=df.index))
    scores["segment"] = seg["labels"]
    scores["predicted_clv"] = clv["pipeline"].predict(df[CLV_NUM + CLV_CAT]).round(2)
    scores.to_csv(PROCESSED / "customer_scores.csv", index=False)
    (REPORTS / "model_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print("saved models/, reports/model_metrics.json, data/processed/customer_scores.csv")
    return report


if __name__ == "__main__":
    run()
