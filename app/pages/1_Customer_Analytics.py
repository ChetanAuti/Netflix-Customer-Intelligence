import app.ui as ui  # noqa: F401
import streamlit as st

from app.ui import bar, setup, sql
from src.database.connection import month_of

setup("Customer Analytics", "📊")

countries = sql("SELECT DISTINCT country FROM customers ORDER BY 1").country.tolist()
sel = st.sidebar.multiselect("Country filter", countries)
cf = "AND c.country IN (" + ",".join(f"'{x}'" for x in sel if x in countries) + ")" if sel else ""

t1, t2, t3 = st.tabs(["Demographics", "Engagement", "Subscriptions"])
with t1:
    a, b = st.columns(2)
    with a:
        st.subheader("Customers by country")
        d = sql(f"SELECT c.country, COUNT(*) customers FROM customers c WHERE 1=1 {cf} GROUP BY c.country ORDER BY customers DESC LIMIT 15")
        bar(d, "country", "customers", horizontal=True)
    with b:
        st.subheader("Age bands")
        d = sql(f"""SELECT CASE WHEN age < 25 THEN '18-24' WHEN age < 35 THEN '25-34' WHEN age < 45 THEN '35-44' WHEN age < 55 THEN '45-54' ELSE '55+' END AS band,
                           COUNT(*) customers FROM customers c WHERE 1=1 {cf} GROUP BY 1 ORDER BY 1""")
        bar(d, "band", "customers")
    st.subheader("Acquisition channel: customers and churn rate")
    d = sql(f"""SELECT c.acquisition_channel AS channel, COUNT(*) customers,
                       ROUND(100.0 * SUM(CASE WHEN cl.churned THEN 1 ELSE 0 END) / COUNT(*), 1) churn_rate_pct
                FROM customers c JOIN churn_labels cl ON cl.customer_id = c.customer_id WHERE 1=1 {cf}
                GROUP BY c.acquisition_channel ORDER BY customers DESC""")
    st.dataframe(d, hide_index=True, use_container_width=True)
with t2:
    a, b = st.columns(2)
    with a:
        st.subheader("Watch hours per month")
        d = sql(f"""SELECT {month_of('v.viewing_date')} AS month, ROUND(SUM(v.watch_duration_minutes) / 60.0) AS watch_hours
                    FROM viewing_activity v JOIN customers c ON c.customer_id = v.customer_id WHERE 1=1 {cf} GROUP BY 1 ORDER BY 1""")
        st.line_chart(d.set_index("month")["watch_hours"], color="#E50914")
    with b:
        st.subheader("Viewing minutes by device")
        d = sql(f"""SELECT v.device_type, SUM(v.watch_duration_minutes) AS minutes FROM viewing_activity v
                    JOIN customers c ON c.customer_id = v.customer_id WHERE 1=1 {cf} GROUP BY 1 ORDER BY 2 DESC""")
        bar(d, "device_type", "minutes")
    st.subheader("Completion rate by genre")
    d = sql(f"""SELECT ct.genre, ROUND(AVG(v.completion_percentage), 1) AS avg_completion_pct, COUNT(*) sessions
                FROM viewing_activity v JOIN content ct ON ct.content_id = v.content_id JOIN customers c ON c.customer_id = v.customer_id
                WHERE 1=1 {cf} GROUP BY ct.genre ORDER BY avg_completion_pct DESC""")
    st.dataframe(d, hide_index=True, use_container_width=True)
with t3:
    d = sql(f"""SELECT sp.plan_name, COUNT(*) active_subscriptions, ROUND(100.0 * SUM(CASE WHEN s.auto_renew THEN 1 ELSE 0 END) / COUNT(*), 1) auto_renew_pct
                FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id JOIN customers c ON c.customer_id = s.customer_id
                WHERE s.subscription_status = 'Active' {cf} GROUP BY sp.plan_name""")
    st.dataframe(d, hide_index=True, use_container_width=True)
    st.subheader("Plan changes (migration paths)")
    d = sql(f"""SELECT a.plan_name AS from_plan, b.plan_name AS to_plan, COUNT(*) AS customers FROM
                (SELECT s.customer_id, s.subscription_start_date, sp.plan_name FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id) a
                JOIN (SELECT s.customer_id, s.subscription_start_date, sp.plan_name FROM subscriptions s JOIN subscription_plans sp ON sp.plan_id = s.plan_id) b
                  ON a.customer_id = b.customer_id AND b.subscription_start_date > a.subscription_start_date
                JOIN customers c ON c.customer_id = a.customer_id
                WHERE a.plan_name <> b.plan_name {cf} GROUP BY 1, 2 ORDER BY 3 DESC""")
    st.dataframe(d, hide_index=True, use_container_width=True)
