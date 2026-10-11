import json

import app.ui as ui  # noqa: F401
import streamlit as st

from app.ui import ROOT, bar, setup, sql

setup("Customer Segmentation", "🧩")

metrics = json.loads((ROOT / "reports" / "model_metrics.json").read_text())["segmentation"]
st.caption(f"K-Means with K = {metrics['chosen_k']} (silhouette-selected; elbow + silhouette plot below).")

summary = sql("""SELECT cs.segment, COUNT(*) AS customers,
                        ROUND(100.0 * SUM(CASE WHEN cl.churned THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct,
                        ROUND(AVG(cs.predicted_clv), 0) AS avg_predicted_clv
                 FROM customer_scores cs JOIN churn_labels cl ON cl.customer_id = cs.customer_id GROUP BY cs.segment ORDER BY customers DESC""")
st.dataframe(summary, hide_index=True, use_container_width=True)
a, b = st.columns(2)
with a:
    st.subheader("Churn rate by segment")
    bar(summary, "segment", "churn_rate_pct")
with b:
    st.subheader("Customers by segment")
    bar(summary, "segment", "customers")

st.subheader("What each segment means")
for name, info in metrics["segments"].items():
    with st.expander(name):
        st.write(info["description"])
        st.json({k: v for k, v in info["profile"].items()})

fig_dir = ROOT / "reports" / "figures"
c1, c2 = st.columns(2)
c1.image(str(fig_dir / "ml_segmentation_pca.png"), caption="Segments projected to 2-D (PCA)")
c2.image(str(fig_dir / "ml_segmentation_elbow_silhouette.png"), caption="Choosing K")

seg = st.selectbox("List customers in a segment", summary.segment.tolist())
if seg in summary.segment.tolist():
    st.dataframe(sql(f"""SELECT cs.customer_id, c.country, cs.churn_probability, cs.risk_tier, cs.predicted_clv FROM customer_scores cs
                         JOIN customers c ON c.customer_id = cs.customer_id WHERE cs.segment = '{seg}' ORDER BY cs.churn_probability DESC LIMIT 100"""),
                 hide_index=True, use_container_width=True)
