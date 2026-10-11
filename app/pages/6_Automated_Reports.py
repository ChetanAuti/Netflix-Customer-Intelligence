from datetime import timedelta

import app.ui as ui  # noqa: F401
import streamlit as st

from app.ui import setup
from src.llm import reports

setup("Automated Reports", "📝")

period = st.radio("Report period", ["weekly", "monthly", "quarterly"], index=1, horizontal=True)
default_end = reports.default_end_date()
end = st.date_input("Period ends on", default_end, max_value=default_end + timedelta(days=30))
st.caption("Facts are computed with SQL; the narrative is clearly separated and marked as AI interpretation.")
if st.button("Generate report", type="primary"):
    with st.spinner("Pulling metrics and drafting..."):
        out = reports.generate_report(period, end)
    st.markdown(out["markdown"])
    st.download_button("Download report (.md)", out["markdown"], file_name=f"{period}_report_{end}.md", mime="text/markdown")
