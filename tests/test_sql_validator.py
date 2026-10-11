import pytest

from src.llm.text_to_sql import SQLValidationError, add_row_cap, extract_sql, validate_sql

GOOD = [
    "SELECT country, COUNT(*) FROM customers GROUP BY country",
    "WITH t AS (SELECT customer_id, SUM(amount) AS s FROM payments WHERE payment_status='Success' GROUP BY customer_id) "
    "SELECT c.country, AVG(t.s) FROM t JOIN customers c ON c.customer_id = t.customer_id GROUP BY c.country",
    "SELECT EXTRACT(YEAR FROM registration_date) AS y, COUNT(*) FROM customers GROUP BY y",
    "SELECT * FROM customer_scores WHERE risk_tier = 'High' LIMIT 10;",
]
BAD = [
    ("DROP TABLE customers", "only SELECT"),
    ("SELECT * FROM customers; DELETE FROM payments", "multiple"),
    ("UPDATE customers SET age = 1", "only SELECT"),
    ("SELECT * FROM customers WHERE 1=1; DROP TABLE x", "multiple"),
    ("SELECT foo FROM customers", "unknown identifier"),
    ("SELECT * FROM passwords", "unknown table"),
    ("SELECT c.nope FROM customers c", "unknown column"),
    ("SELECT * INTO backup FROM customers", "forbidden"),
    ("", "empty"),
    ("WITH x AS (SELECT 1) DELETE FROM customers", "forbidden"),
]


@pytest.mark.parametrize("sql", GOOD)
def test_good_queries_pass(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize("sql,msg", BAD)
def test_bad_queries_blocked(sql, msg):
    with pytest.raises(SQLValidationError, match=msg):
        validate_sql(sql)


def test_semicolon_inside_string_literal_is_allowed():
    assert validate_sql("SELECT * FROM customers WHERE city = 'a;b'")


def test_row_cap_wraps_query():
    assert "LIMIT 500" in add_row_cap("SELECT 1")


def test_extract_sql_from_fenced_block():
    assert extract_sql("here:\n```sql\nSELECT 1\n```") == "SELECT 1"


def test_all_37_project_queries_validate():
    import re
    from pathlib import Path
    sql = (Path(__file__).resolve().parents[1] / "database" / "sql_business_questions.sql").read_text()
    parts = re.split(r"\n-- Q(\d+)\. ", sql)[1:]
    assert len(parts) // 2 == 37
    for i in range(0, len(parts), 2):
        body = parts[i + 1].split("\n", 1)[1]
        body = re.sub(r"\n-- =+\n-- SECTION.*", "", body, flags=re.S)
        for stmt in [x for x in body.split(";\n\n") if x.strip()]:
            validate_sql(re.sub(r"--[^\n]*", "", stmt))
