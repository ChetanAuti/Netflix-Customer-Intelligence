import pytest

from src.llm import explainer, rag, rca, reports, retention, text_to_sql


def test_rag_reports_record_counts_and_finds_topic():
    out = rag.answer("payment failed charged")
    assert out["n_records"] > 0 and "Based on" in out["answer"]
    assert out["records"].text.str.contains("payment|charged", case=False).any()


def test_rag_no_result_message():
    assert rag.answer("zzzxqv qqqwww")["n_records"] == 0


def test_text_to_sql_offline_template_runs_and_is_audited():
    r = text_to_sql.ask("Which country has the highest churn rate?")
    assert r.status == "ok" and len(r.df) > 0 and r.answer


def test_text_to_sql_unknown_question_offline_is_graceful():
    r = text_to_sql.ask("compose a sonnet about autumn")
    assert r.status in {"no_llm", "ok"}


def test_number_guard_flags_invented_numbers():
    ev = {"steps": {"a": {"x": 12.5, "n": 40}}}
    assert rca.unverified_numbers("Churn was 12.5% across 40 customers", ev) == []
    assert rca.unverified_numbers("Churn was 99.1% across 40 customers", ev) == ["99.1"]


def test_rca_rejects_bad_month_and_unknown_country():
    with pytest.raises(ValueError):
        rca.run("India", "2026-13")
    with pytest.raises(ValueError):
        rca.investigate("Narnia' OR '1'='1", "2026-05")


def test_rca_evidence_shape():
    out = rca.run("India", "2026-05")
    assert {"1_churn", "4_support"} <= set(out["evidence"]["steps"]) and out["text"]


def test_report_has_facts_and_ai_section():
    r = reports.generate_report("weekly")
    assert "Calculated facts" in r["markdown"] and "AI interpretation" in r["markdown"]


def test_explainer_and_retention_for_a_high_risk_customer():
    import pandas as pd
    from src.utils.config import PROCESSED
    cid = pd.read_csv(PROCESSED / "customer_scores.csv").query("risk_tier == 'High'").customer_id.iloc[0]
    e = explainer.explain_customer(cid)
    assert cid in e["explanation"] and len(e["drivers"]) == 5
    assert retention.recommend(cid)["actions"]
