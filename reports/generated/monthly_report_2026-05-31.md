# Monthly Business Report - 2026-05-02 to 2026-05-31

## Calculated facts (SQL)
| metric | current | previous | change |
|---|---|---|---|
| new_customers | 79 | 81 | -2.5% |
| cancellations | 237 | 217 | +9.2% |
| churn_rate_pct | 3.69 | 3.3 | +11.8% |
| revenue_usd | 16794.25 | 16630.21 | +1.0% |
| payment_failure_rate_pct | 6.49 | 7.25 | -10.5% |
| active_viewers | 4243 | 3185 | +33.2% |
| watch_hours | 7011.1 | 3577.7 | +96.0% |
| tickets | 277 | 253 | +9.5% |
| avg_csat | 3.2 | 3.32 | -3.6% |
| avg_feedback_rating | 3.26 | 3.15 | +3.5% |

Top complaints this period:
- Buffering: 25 tickets
- Low resolution: 20 tickets
- Invoice question: 20 tickets

High-risk active customers (churn model): 735

_Data available through: {"payments": "2026-05-31", "viewing": "2026-06-30", "tickets": "2026-06-30", "churn": "2026-06-30", "feedback": "2026-06-30"}_

## AI interpretation
_Written by rule-based templates (no API key configured) from the facts above; it is interpretation, not measured data._

**Executive summary**
79 customers joined and 237 cancelled in 2026-05-02 to 2026-05-31, a churn rate of 3.69% of the opening base of 6,426. Revenue was $16,794 across 1,310 payment attempts.

**Key findings**
- Revenue rose 1.0% versus the previous monthly period
- Churn rate rose 11.8% versus the previous monthly period
- Active viewers rose 33.2% versus the previous monthly period
- The payment failure rate fell 10.5% versus the previous monthly period
- Support tickets rose 9.5% versus the previous monthly period

**Risks**
- Churn is accelerating (237 cancellations, 3.69% of the opening base).
- Support satisfaction slipped to 3.2/5.
- 735 active customers are currently in the High churn-risk tier.

**Recommendations**
- Run the payment-recovery flow for customers with recent failed payments.
- Target the High-risk, high-value customers with personalised outreach (see Retention page).
- Investigate the top complaint driver: 'Buffering'.
