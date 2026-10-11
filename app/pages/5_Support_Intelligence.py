import app.ui as ui  # noqa: F401
import streamlit as st

from app.ui import bar, setup, sql
from src.database.connection import quarter_of
from src.llm import rag

setup("Support Intelligence", "🎧")

q = st.text_input("Search complaints and feedback", "billing refund after cancellation")
if q:
    out = rag.answer(q)
    st.info(out["answer"])
    st.caption(f"Based on {out['n_records']:,} records ({out['n_distinct']} distinct texts)")

a, b = st.columns(2)
with a:
    st.subheader("Tickets per quarter")
    d = sql(f"""SELECT {quarter_of('ticket_date')} AS quarter, COUNT(*) AS tickets FROM support_tickets WHERE ticket_date IS NOT NULL GROUP BY 1 ORDER BY 1""")
    st.line_chart(d.set_index("quarter")["tickets"], color="#E50914")
with b:
    st.subheader("Sentiment of feedback (rating-based labels)")
    d = sql("SELECT sentiment_label, COUNT(*) AS feedback FROM customer_feedback GROUP BY sentiment_label")
    bar(d, "sentiment_label", "feedback")

st.subheader("Complaint categories: volume, resolution time, CSAT")
d = sql("""SELECT issue_category, COUNT(*) tickets, ROUND(AVG(resolution_time_hours), 1) avg_resolution_hours,
                  ROUND(AVG(customer_satisfaction_score), 2) avg_csat FROM support_tickets GROUP BY issue_category ORDER BY tickets DESC""")
st.dataframe(d, hide_index=True, use_container_width=True)

st.subheader("Top sub-categories")
d = sql("""SELECT issue_subcategory, COUNT(*) tickets, ROUND(AVG(customer_satisfaction_score), 2) avg_csat
           FROM support_tickets GROUP BY issue_subcategory ORDER BY tickets DESC LIMIT 12""")
st.dataframe(d, hide_index=True, use_container_width=True)
