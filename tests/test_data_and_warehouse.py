import pandas as pd
import pytest

from couponroi import data, warehouse
from couponroi.synthetic import make_tables


def test_synthetic_tables_pass_validation():
    issues = data.validate(make_tables(n_households=80, seed=1))
    assert not [i for i in issues if i.severity == "error"]


def test_validation_catches_missing_column_and_duplicates():
    t = make_tables(n_households=60, seed=2)
    t["products"] = t["products"].drop(columns=["department"])
    t["demographics"] = pd.concat([t["demographics"], t["demographics"].head(1)])
    errors = {(i.table, i.check) for i in data.validate(t) if i.severity == "error"}
    assert ("products", "columns") in errors
    assert ("demographics", "unique") in errors


def test_fetch_refuses_broken_data(tmp_path, monkeypatch):
    def broken(**_):
        t = make_tables(n_households=40, seed=3)
        t["campaign_descriptions"].loc[0, "end_date"] = pd.Timestamp("2000-01-01")
        return t

    monkeypatch.setattr(data, "load_source", broken)
    with pytest.raises(ValueError, match="failed validation"):
        data.fetch(raw_dir=tmp_path)


def test_warehouse_quality_checks_pass(con):
    checks = {c["check"]: c for c in warehouse.quality_checks(con)}
    failed = [k for k, c in checks.items() if not c["passed"]]
    assert not failed, failed


def test_warehouse_connections_are_single_threaded(synth_dirs):
    # parallel float sums differ in the last bit from run to run (decision log 14). The synthetic
    # data is too small to show it, so check the setting itself
    con = warehouse.connect(synth_dirs["db"])
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    con.close()


def test_panel_is_dense_and_reconciles(con):
    hh = con.execute("SELECT count(*) FROM dim_household").fetchone()[0]
    weeks = con.execute("SELECT count(*) FROM dim_week").fetchone()[0]
    assert con.execute("SELECT count(*) FROM hh_week").fetchone()[0] == hh * weeks
    raw = con.execute("SELECT sum(sales_value) FROM raw_transactions").fetchone()[0]
    panel = con.execute("SELECT sum(spend) FROM hh_week").fetchone()[0]
    assert abs(raw - panel) < 0.01


def test_local_time_conversion_matches_week_numbers(con):
    """Every transaction's local (US Eastern, DST-aware) date must fall inside its week."""
    bad = con.execute(
        """SELECT count(*) FROM fct_transactions t JOIN dim_week w USING (week)
           WHERE t.date_local NOT BETWEEN w.week_start AND w.week_end"""
    ).fetchone()[0]
    assert bad == 0


def test_week_of_macro(con):
    assert con.execute("SELECT week_of(DATE '2017-01-02')").fetchone()[0] == 2
    assert con.execute("SELECT week_of(DATE '2017-01-01')").fetchone()[0] == 1
    assert con.execute("SELECT week_of(DATE '2016-12-25')").fetchone()[0] == 0  # floor, not truncation
    assert con.execute("SELECT week_of(DATE '2017-12-31')").fetchone()[0] == 53
