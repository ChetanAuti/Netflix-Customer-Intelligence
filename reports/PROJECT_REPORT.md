# Project Report - Netflix Customer Intelligence & AI Analytics Copilot

## 1. Data quality (what was wrong and what we did)
All decisions are logged in `reports/logs/cleaning_log.json`.

| Issue | Count | Treatment |
|---|---|---|
| Duplicate customers (exact copies) | 48 | dropped |
| Country missing | 409 | imputed from the customer's most common login country (all 409 recovered) |
| Country / device spelled inconsistently | many (10,285 device values) | mapped to one canonical spelling |
| Age implausible (<13 or >100) / missing | 21 / 321 | set to missing, imputed with median of customer type; `age_imputed` flag kept |
| City / language missing | 483 / 251 | "Unknown" / most common language in the country |
| Watch / session minutes outliers | 161 | capped at Q3 + 3xIQR (163 / 178 min); flagged |
| Session shorter than watched time | 6,068 | session raised to watched time |
| Completion % missing | 531 | median of the title |
| Payment amounts that are not a catalogue price (0, 13.98, 76.89, 252.89 ...) | 1,112 | replaced by the price of the subscription segment; flagged |
| `subscription_id` absent from all 110,735 payments | 110,735 | resolved by joining on customer + the segment active on the payment date |
| Invalid ticket dates | 61 | set to NULL, flagged (kept in CSAT/volume statistics) |
| Sentiment label contradicts rating | 263 | re-derived from rating (1-2 Negative, 3 Neutral, 4-5 Positive) |
| CSAT / resolution time empty for open tickets | 1,528 | left NULL on purpose (not yet resolved) |

Referential integrity: 0 orphan rows in any table after cleaning.

## 2. What the data says (EDA, 24 figures in `reports/figures/eda_*.png`)
* Overall churn is **24.9%** (1,991 of 8,000). Cancellations run every month from Oct-2025 to Jun-2026.
* Top stated churn reasons: *Too expensive* (365), *Found alternative platform* (284), *Not enough content* (217), *No longer needed* (212), *Technical issues* (182).
* **Engagement tier is the cleanest churn signal:** Low-engagement customers churn 33.9%, Medium 25.8%, High 16.1%.
* **Failed payments:** customers with any failed payment in the trailing 180 days churn 38.0% vs 24.3% without.
* **Auto-renew off:** 28.7% churn vs 24.0% with auto-renew on.
* Tenure is *not* a strong protector here (22-30% across buckets).
* Support complaints are dominated by templated billing/refund/login/buffering themes (the corpus has only 32 distinct ticket texts and 13 distinct feedback texts), so text analytics find clear themes but little nuance.

## 3. Models

### 3.1 Churn classification
* Features: 40 numeric + 5 categorical, all measured at one snapshot (end of data) with trailing windows. No outcome field (status, cancellation, churn reason/date) is an input.
* Candidates tuned with 5-fold CV grid search; class imbalance via class weights; threshold chosen on out-of-fold data only.

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic regression | 0.645 | 0.623 |
| Random forest | 0.640 | 0.615 |
| **Gradient boosting (selected)** | **0.647** | **0.617** |

At the F1-optimal threshold (0.45): accuracy 51%, precision 30%, recall 73%, F1 0.43. Confusion matrix on 1,600 test customers: 528 TN, 674 FP, 107 FN, 291 TP.

**Business use is ranking, not classification.** Predicted-risk deciles on the test set: top decile 1.46x the base churn rate, bottom decile 0.65x. The three risk tiers (top 15% / next 25% / rest) separate churn at 38.8% / 30.8% / 18.9%.

Top permutation importances: tenure, total watch minutes, support satisfaction, engagement score, watch-time trend, days since last activity.

### 3.2 Why not a higher AUC? (important)
* Our **first** version used each churned customer's own cancellation date as the observation cutoff and scored AUC 0.99. That was **leakage**: the cutoff itself encodes the label (window length differs between churned and active customers). It was removed.
* A strictly forward-looking test (`leakage_check.py`: features from before a monthly snapshot, label = cancels within 30 / 90 days, customer-grouped CV) gives **AUC ~0.56**. In this dataset churn is explained by behaviour measured over the whole period much more than by what customers did 1-3 months before leaving, so true early-warning power is limited.
* The deployed score is therefore best described as a *churn-propensity / risk-ranking* score based on current behaviour. Scores shown for existing customers are out-of-fold (each customer scored by a model that did not train on them).
* Do not quote "predicts churn 30 days ahead" - the evidence does not support it.

### 3.3 Segmentation (K-Means, K = 5; silhouette ~0.20)
| Segment | Customers | Churn | Avg CLV |
|---|---|---|---|
| High-Value Loyal | 3,011 | 17.8% | $172 |
| Fading Binge Viewers | 2,338 | 26.3% | $185 |
| At-Risk Low-Engagement | 1,684 | 33.7% | $188 |
| Support-Heavy Satisfied Users | 632 | 22.5% | $156 |
| Billing-Troubled New Signups | 335 | 38.2% | $90 |

Silhouette is low (~0.20 for every K): the customer base is a continuum, not distinct clusters. Segments are useful as a *labelling device* (churn rates differ by 2x) rather than natural groups.

### 3.4 CLV regression
Gradient boosting: R2 0.979, MAE $9.61, RMSE $13.36 (mean CLV $175). Inputs exclude every payment-derived field. The high R2 is mostly arithmetic: lifetime value ~ tenure x plan price, so this model mostly confirms that, rather than discovering hidden value drivers.

## 4. AI copilot
| Capability | How it works | Guardrails |
|---|---|---|
| Text-to-SQL | LLM sees schema, writes one SELECT; offline mode uses 10 templates | SELECT-only, 16 blocked keywords, single statement, table + column + identifier check against the real schema, 500-row cap, 15 s timeout, read-only session, audit log. The 37 project queries all pass the validator (tested) |
| RAG | TF-IDF over tickets + feedback (Chroma optional); identical texts are collapsed and counted | every answer states the number of records it rests on |
| Prediction explainer | one-feature-substitution drivers + customer facts -> plain language | LLM may use only the supplied facts |
| Report generator | weekly / monthly / quarterly; facts via SQL, narrative separate | narrative labelled "AI interpretation"; rule-based fallback |
| Root-cause agent | fixed 5-step SQL investigation (churn delta, engagement, payments, support, plan/reason mix) | every number in the explanation must appear in the evidence, otherwise the text is rejected and a rules-only version is shown; country is whitelisted (no injection) |
| Retention advice | transparent rules (risk tier x value x trigger) first, LLM wording second | no offer for Low-risk customers |

## 5. Business recommendations
1. **Fix involuntary churn first.** Failed-payment customers churn at 38% vs 24%. Add card-retry and "update payment method" reminders; the churn model's High tier already contains 734 active customers (about $9.1k MRR).
2. **Run a low-engagement win-back program.** The At-Risk Low-Engagement segment (1,684 customers) churns 34%; trigger recommendations / re-engagement when watch time falls or a customer is silent for 30+ days.
3. **Address price perception** (top stated reason): test a lower-priced ad tier or annual-plan discounts for price-sensitive segments rather than blanket discounts.
4. **Tier the spend.** Personal outreach only for High-risk *and* high-CLV customers; automated emails for High-risk low-CLV; nothing for Low risk.
5. **Improve measurement:** keep monthly snapshots of customer behaviour. Genuine early-warning modelling needs history this extract does not contain.

## 6. Limitations
* Churn signal is weak and mostly contemporaneous (see 3.2); the model ranks risk, it does not give a reliable early warning.
* Viewing data covers only Jul-2025 to Jun-2026; payments end 2026-05-31.
* Ticket and feedback texts are heavily templated (32 / 13 distinct texts), limiting NLP value.
* PostgreSQL loading, the 37 queries on PostgreSQL, the real Streamlit server, and LLM mode were not run in the build environment (see README).
* Q-level SQL corrections vs the first draft: Q8, Q24, Q28, Q31, Q34, Q35, Q37 now anchor on the data's last date instead of `CURRENT_DATE` (the data ends 2026-06-30); Q24 uses the opening active base as denominator; Q28 uses the model's High tier; Q34 uses consecutive plan changes and compares upgrader vs downgrader churn.
