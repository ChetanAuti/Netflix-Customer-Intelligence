import app.ui as ui  # noqa: F401
import pandas as pd
import streamlit as st

from app.ui import bar, setup, sql
from src.llm import explainer, retention

setup("Churn Prediction", "⚠️")

top = sql("""SELECT cs.customer_id, cs.churn_probability, cs.risk_tier, cs.segment, s.monthly_price,
                    ROUND(cs.churn_probability * s.monthly_price * 12, 2) AS at_risk_annual_revenue
             FROM customer_scores cs JOIN churn_labels cl ON cl.customer_id = cs.customer_id
             JOIN subscriptions s ON s.customer_id = cs.customer_id AND s.subscription_status = 'Active'
             WHERE cl.churned = FALSE ORDER BY at_risk_annual_revenue DESC LIMIT 200""")
st.subheader("Highest at-risk revenue (active customers)")
st.dataframe(top.head(25), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Inspect one customer")
choice = st.selectbox("Pick from the top at-risk list", top.customer_id.tolist())
typed = st.text_input("...or type any customer id (e.g. CUST000123)")
cid = typed.strip() or choice

if cid:
    try:
        res = explainer.explain_customer(cid)
    except KeyError:
        st.error(f"Customer {cid} not found.")
    else:
        f = res["facts"]
        a, b, c, d = st.columns(4)
        a.metric("Churn probability", f"{f['churn_probability_pct']}%")
        b.metric("Risk tier", f["risk_tier"])
        c.metric("Segment", f["segment"])
        d.metric("Predicted CLV", f"${f['predicted_clv_usd']:.0f}")
        st.markdown("#### Why? (AI explanation)")
        st.write(res["explanation"])
        drivers = pd.DataFrame(res["drivers"])
        drivers["factor"] = drivers.feature.map(lambda x: explainer.FRIENDLY.get(x, x))
        st.markdown("#### Model risk drivers (positive = pushes risk up, percentage points)")
        bar(drivers, "factor", "risk_effect_pts", horizontal=True)
        with st.expander("Customer facts used"):
            st.json(f)
        st.markdown("#### Recommended retention actions")
        rec = retention.recommend(cid)
        for act in rec["actions"]:
            st.markdown(f"- **{act['action']}** - {act['why']}")
        if rec["plan_text"]:
            st.write(rec["plan_text"])
