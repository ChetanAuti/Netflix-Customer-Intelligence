import json

import app.ui as ui  # noqa: F401
import pandas as pd
import streamlit as st

from app.ui import ROOT, setup
from src.llm import rag, rca, text_to_sql

setup("AI Analytics Copilot", "🤖")

tab_sql, tab_rag, tab_rca = st.tabs(["Ask the data (text-to-SQL)", "Ask the customer voice (RAG)", "Why did churn change? (root cause)"])

EXAMPLES = ["Which country has the highest churn rate?", "What is the revenue by plan?", "What are the top 10 most watched titles?",
            "What are the most common failed payment reasons?", "How many high risk customers are in each segment?"]

with tab_sql:
    st.write("Questions become a validated, read-only SQL query. Try an example or write your own.")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex{i}"):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q")
    if st.button("Ask", type="primary") and q:
        with st.spinner("Generating and running SQL..."):
            r = text_to_sql.ask(q)
        if r.status == "ok":
            st.markdown("**Generated SQL** (validated: SELECT-only, schema-checked, 500-row cap, 15 s timeout)")
            st.code(r.sql, language="sql")
            st.dataframe(r.df, hide_index=True, use_container_width=True)
            if r.df.shape[1] == 2 and pd.api.types.is_numeric_dtype(r.df.iloc[:, 1]) and 1 < len(r.df) <= 60:
                st.bar_chart(r.df.set_index(r.df.columns[0]).iloc[:, 0])
            st.success(r.answer)
        elif r.status == "blocked":
            st.error(f"Query blocked by the validation layer: {r.error}")
            st.code(r.sql, language="sql")
        else:
            st.warning(r.error)
    log = ROOT / "reports" / "logs" / "query_audit.jsonl"
    if log.exists():
        with st.expander("Audit log (last 10 queries)"):
            rows = [json.loads(x) for x in log.read_text().splitlines()[-10:]]
            st.dataframe(pd.DataFrame(rows)[::-1], hide_index=True, use_container_width=True)

with tab_rag:
    rq = st.text_input("Ask about what customers are saying", "What are customers saying about payment failures?")
    src = st.radio("Source", ["Both", "Support tickets", "Feedback"], horizontal=True)
    if st.button("Search", key="rag_go"):
        s = {"Both": None, "Support tickets": "support_ticket", "Feedback": "feedback"}[src]
        out = rag.answer(rq, source=s)
        st.info(out["answer"])
        st.caption(f"Grounded in {out['n_records']:,} records ({out['n_distinct']} distinct texts).")
        if out["n_records"]:
            st.dataframe(out["records"][["source", "n_records", "category", "text", "similarity"]], hide_index=True, use_container_width=True)

with tab_rca:
    st.write("A fixed investigation plan runs SQL checks (churn delta, engagement, payments, support, plan/reason mix). "
             "Numbers in the explanation are verified against the query results.")
    default_country, default_month = rca.pick_target()
    countries = rca.countries()
    c1, c2 = st.columns(2)
    country = c1.selectbox("Country", countries, index=countries.index(default_country) if default_country in countries else 0)
    month = c2.text_input("Month (YYYY-MM)", default_month)
    if st.button("Investigate"):
        try:
            out = rca.run(country, month)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.write(out["text"])
            st.caption(f"Explanation source: {out['source']}")
            with st.expander("Evidence (raw query results)"):
                st.json(out["evidence"]["steps"])
