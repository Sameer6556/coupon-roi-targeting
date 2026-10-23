"""Build the DuckDB warehouse from the raw parquet files by running sql/*.sql in order.

DuckDB runs in-process, needs no server, and reads parquet directly, so the whole
SQL layer is reproducible on a laptop. The SQL files are the source of truth for the
star schema; Python only wires them up and checks the result.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import duckdb

from . import config

log = logging.getLogger(__name__)


def connect(path: Path = config.WAREHOUSE_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if read_only and not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `couponroi build` first.")
    return duckdb.connect(str(path), read_only=read_only)


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def create_raw_views(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    for name in config.RAW_TABLES:
        p = raw_dir / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run `couponroi data` first.")
        # views (not tables) so the raw files are not copied into the warehouse twice
        con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM read_parquet({_sql_literal(p)})")


def build(
    raw_dir: Path = config.RAW_DIR,
    db_path: Path = config.WAREHOUSE_PATH,
    sql_dir: Path = config.SQL_DIR,
    report_dir: Path | None = config.RESULTS_DIR,
) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        create_raw_views(con, raw_dir)
        timings = {}
        for f in sorted(sql_dir.glob("[0-9][0-9]_*.sql")):
            t0 = time.perf_counter()
            con.execute(f.read_text())
            timings[f.name] = round(time.perf_counter() - t0, 2)
            log.info("ran %s in %.2fs", f.name, timings[f.name])
        checks = quality_checks(con)
    finally:
        con.close()
    failed = [c for c in checks if not c["passed"]]
    report = {"timings_s": timings, "checks": checks}
    if report_dir is not None:  # tests build throwaway warehouses and pass None
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    if failed:
        raise AssertionError(f"warehouse checks failed: {failed}")
    return report


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def quality_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """dbt-style tests on the built tables. Each returns a value and a pass/fail."""
    checks = []

    def check(name: str, value, passed: bool) -> None:
        checks.append({"check": name, "value": value, "passed": bool(passed)})

    raw_rows = _one(con, "SELECT count(*) FROM raw_transactions")
    fct_rows = _one(con, "SELECT count(*) FROM fct_transactions")
    check("fct_transactions keeps every raw line", {"raw": raw_rows, "fct": fct_rows}, raw_rows == fct_rows)

    raw_sales = _one(con, "SELECT round(sum(sales_value), 2) FROM raw_transactions")
    panel_sales = _one(con, "SELECT round(sum(spend), 2) FROM hh_week")
    check(
        "panel spend reconciles to raw sales",
        {"raw": raw_sales, "panel": panel_sales},
        abs(raw_sales - panel_sales) < 0.01,
    )

    hh = _one(con, "SELECT count(*) FROM dim_household")
    weeks = _one(con, "SELECT count(*) FROM dim_week")
    panel = _one(con, "SELECT count(*) FROM hh_week")
    check(
        "panel is dense (households x weeks)",
        {"households": hh, "weeks": weeks, "rows": panel},
        panel == hh * weeks,
    )

    dup = _one(
        con, "SELECT count(*) FROM (SELECT household_id, week FROM hh_week GROUP BY ALL HAVING count(*) > 1)"
    )
    check("panel has one row per household-week", dup, dup == 0)

    # week 1 is one Sunday; week 53 lost Christmas Day. Every other week must be 7 days,
    # which is what proves the UTC -> US Eastern conversion lines up with the week numbers.
    gap = _one(con, "SELECT count(*) FROM dim_week WHERE week BETWEEN 2 AND 52 AND days_observed <> 7")
    check("weeks 2-52 have exactly 7 local days", gap, gap == 0)

    orphan_exposure = _one(
        con, "SELECT count(*) FROM fct_campaign_exposure ANTI JOIN dim_household USING (household_id)"
    )
    check("every exposed household shopped", orphan_exposure, orphan_exposure == 0)

    bad_redemptions = _one(
        con,
        """SELECT count(*) FROM fct_redemptions r JOIN dim_campaign c USING (campaign_id)
           WHERE r.redemption_date NOT BETWEEN c.start_date AND c.end_date""",
    )
    check("redemptions fall inside their campaign window", bad_redemptions, bad_redemptions == 0)

    redeemed_unexposed = _one(
        con,
        "SELECT count(*) FROM fct_redemptions ANTI JOIN fct_campaign_exposure USING (household_id, campaign_id)",
    )
    # not a hard failure: worth knowing, and it is 0 in the real data
    check("only exposed households redeem", redeemed_unexposed, True)

    demo_share = _one(con, "SELECT avg(CAST(has_demographics AS INTEGER)) FROM dim_household")
    check("share of households with demographics", round(demo_share, 3), True)
    return checks
