"""Executive Dashboard (entry point):  streamlit run app/dashboard.py"""
import app.ui as ui  # noqa: F401  (sets sys.path)
import streamlit as st

from app.ui import bar, setup, sql

setup("Executive Dashboard")

k = sql("""SELECT
  (SELECT COUNT(*) FROM customers) total,
  (SELECT COUNT(*) FROM churn_labels WHERE churned = FALSE) active,
  (SELECT ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / COUNT(*), 1) FROM churn_labels) churn_rate,
  (SELECT SUM(monthly_price) FROM subscriptions WHERE subscription_status = 'Active') mrr,
  (SELECT COUNT(*) FROM customer_scores cs JOIN churn_labels cl ON cl.customer_id = cs.customer_id
     WHERE cs.risk_tier = 'High' AND cl.churned = FALSE) high_risk,
  (SELECT ROUND(AVG(customer_satisfaction_score), 2) FROM support_tickets) csat""").iloc[0]

c = st.columns(6)
c[0].metric("Total customers", f"{int(k.total):,}")
c[1].metric("Active customers", f"{int(k.active):,}")
c[2].metric("Churn rate", f"{k.churn_rate}%")
c[3].metric("Monthly recurring revenue", f"${k.mrr:,.0f}")
c[4].metric("High-risk active customers", f"{int(k.high_risk):,}")
c[5].metric("Avg support CSAT", f"{k.csat}/5")

left, right = st.columns(2)
with left:
    st.subheader("Active subscriptions & revenue by plan")
    plan = sql("""SELECT sp.plan_name, COUNT(*) AS subscriptions, SUM(s.monthly_price) AS monthly_revenue
                  FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id
                  WHERE s.subscription_status = 'Active' GROUP BY sp.plan_name ORDER BY monthly_revenue DESC""")
    st.dataframe(plan, hide_index=True, use_container_width=True)
    bar(plan, "plan_name", "monthly_revenue")
with right:
    st.subheader("Cancellations per month")
    from src.database.connection import month_of
    canc = sql(f"""SELECT {month_of('churn_date')} AS month, COUNT(*) AS cancellations FROM churn_labels
                   WHERE churned = TRUE GROUP BY 1 ORDER BY 1""")
    st.line_chart(canc.set_index("month")["cancellations"], color="#E50914")

l2, r2 = st.columns(2)
with l2:
    st.subheader("Churn risk tiers (active customers)")
    tiers = sql("""SELECT cs.risk_tier, COUNT(*) AS customers FROM customer_scores cs JOIN churn_labels cl ON cl.customer_id = cs.customer_id
                   WHERE cl.churned = FALSE GROUP BY cs.risk_tier""")
    bar(tiers, "risk_tier", "customers")
with r2:
    st.subheader("Top churn reasons")
    reasons = sql("SELECT churn_reason, COUNT(*) AS customers FROM churn_labels WHERE churned = TRUE GROUP BY churn_reason ORDER BY customers DESC LIMIT 8")
    bar(reasons, "churn_reason", "customers", horizontal=True)

st.caption("Risk tiers come from the churn model's out-of-fold scores (top 15% = High). Source: PostgreSQL / SQLite via DATABASE_URL.")
