import duckdb
import pandas as pd
import pytest

from couponroi import ask as A

BLOCKED = [
    "DROP TABLE households",
    "SELECT * FROM households; DELETE FROM households",
    "INSERT INTO households VALUES (1)",
    "UPDATE households SET total_spend = 0",
    "COPY households TO 'out.csv'",
    "ATTACH 'other.db'",
    "PRAGMA database_list",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
    "SELECT * FROM duckdb_settings()",
    "SELECT * FROM range(10)",
    "SELECT * FROM secret_table",
    "SELECT * FROM information_schema.tables",
    "SELECT current_setting('threads')",
    "this is not sql",
]


@pytest.mark.parametrize("sql", BLOCKED)
def test_guard_blocks(sql):
    with pytest.raises(A.UnsafeSQL):
        A.guard(sql)


def test_guard_allows_selects_and_ctes_and_forces_a_limit():
    assert A.guard("select count(*) from households").endswith("LIMIT 500")
    out = A.guard(
        "with x as (select * from household_week) select week, sum(spend) from x group by week limit 10"
    )
    assert "LIMIT 10" in out
    assert A.guard("select * from campaigns limit 100000").endswith("LIMIT 500")


def test_extract_sql_from_markdown():
    assert A.extract_sql("Here you go:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert A.extract_sql("SELECT 2") == "SELECT 2"


def test_results_match_ignores_order_and_names():
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = pd.DataFrame({"n": [2, 1], "label": ["b", "a"]})
    assert A.results_match(a, b)
    assert not A.results_match(a, pd.DataFrame({"x": [1, 3], "y": ["a", "b"]}))


class FakeClient:
    """Pretends to be an LLM: answers each eval question with its reference SQL."""

    class cfg:
        model = "fake"

    def __init__(self, answers):
        self.answers = answers

    def complete(self, messages, temperature=0.0):
        return f"```sql\n{self.answers[messages[-1]['content']]}\n```"


def test_eval_harness_scores_a_perfect_client_at_100():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE households AS SELECT range AS household_id, range * 10.0 AS total_spend FROM range(5)"
    )
    items = [
        {"id": 1, "question": "how many households?", "sql": "SELECT count(*) FROM households"},
        {
            "id": 2,
            "question": "top spender?",
            "sql": "SELECT household_id FROM households ORDER BY total_spend DESC LIMIT 1",
        },
    ]
    res = A.evaluate(con, FakeClient({i["question"]: i["sql"] for i in items}), items)
    assert res["execution_accuracy"] == 1.0


def test_readonly_connection_blocks_file_access(tmp_path):
    db = tmp_path / "x.duckdb"
    duckdb.connect(str(db)).execute("CREATE TABLE households AS SELECT 1 AS household_id").close()
    con = A.connect_readonly(db)
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('/etc/hostname')").df()
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE y AS SELECT 1")


def test_eval_set_is_well_formed():
    items = A.load_eval()
    assert len(items) == 20
    for it in items:
        A.guard(it["sql"])  # every reference query passes the guard
