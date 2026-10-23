"""Ask the data in plain English: question -> one read-only SQL query -> table -> short summary.

Guardrails, in the order they apply:
  1. The model only sees a small set of documented tables (the app bundle).
  2. The SQL is parsed with sqlglot: exactly one SELECT, only whitelisted tables, no
     file/URL reading functions, and a LIMIT is forced on.
  3. DuckDB is opened read-only with external access disabled, so even a query that slipped
     through could not write anything or read files or URLs.
  4. The summary prompt only gets the result table and is told to use only its numbers.

Any OpenAI-compatible endpoint works (Ollama locally, Groq, Gemini, OpenAI, ...):
  COUPONROI_LLM_BASE_URL   default http://localhost:11434/v1  (Ollama)
  COUPONROI_LLM_MODEL      default qwen2.5-coder:7b
  COUPONROI_LLM_API_KEY    optional
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from . import config

APP_DB = config.APP_DIR / "app.duckdb"
EVAL_PATH = config.EVAL_DIR / "ask_eval.jsonl"
MAX_ROWS = 500

SCHEMA: dict[str, dict] = {
    "household_week": {
        "about": "one row per household per retailer week (dense: weeks with no shopping have zeros)",
        "columns": {
            "household_id": "household key",
            "week": "retailer week number 1-53 (2017); weeks run Monday-Sunday",
            "week_start": "date of the Monday that starts the week",
            "spend": "total spend in USD including fuel",
            "spend_ex_fuel": "spend in USD excluding fuel (the main spend measure)",
            "trips": "number of shopping trips (baskets)",
            "departments_shopped": "number of departments bought from",
            "private_label_spend": "spend on store-brand items, USD",
            "promo_spend": "spend on items on display or in the weekly mailer, USD",
            "discount_amount": "total discounts received, USD",
            "coupon_redemptions": "campaign coupons redeemed that week",
            "active_campaigns": "number of campaigns the household had in hand that week",
            "is_active": "true if the household shopped that week",
        },
    },
    "households": {
        "about": "one row per household. Demographic columns are NULL for the ~68% without demographics",
        "columns": {
            "household_id": "household key",
            "first_week": "first week with a purchase",
            "last_week": "last week with a purchase",
            "active_weeks": "number of weeks with a purchase",
            "baskets": "number of shopping trips in the year",
            "total_spend": "total spend in the year, USD, incl. fuel",
            "total_spend_ex_fuel": "total spend in the year, USD, excl. fuel",
            "campaigns_received": "number of campaigns sent to the household",
            "type_a_received": "number of Type A (personalised booklet) campaigns received",
            "redemptions": "coupons redeemed in the year",
            "has_demographics": "true if demographic columns are filled",
            "age": "age band, e.g. '45-54'",
            "income": "income band, e.g. '50-74K'",
            "home_ownership": "e.g. 'Homeowner', 'Renter'",
            "marital_status": "'Married' or 'Unmarried'",
            "household_size": "'1' to '5+'",
            "household_comp": "e.g. '2 Adults Kids'",
            "kids_count": "'0' to '3+'",
        },
    },
    "campaigns": {
        "about": "one row per marketing campaign (27 campaigns)",
        "columns": {
            "campaign_id": "campaign key",
            "campaign_type": "'Type A' (personalised booklets), 'Type B' or 'Type C'",
            "start_date": "first day",
            "end_date": "last day",
            "start_week": "retailer week the campaign started (can be <= 0 for 2016 starts)",
            "end_week": "retailer week the campaign ended (can be > 53)",
            "duration_weeks": "length in weeks",
            "households_targeted": "households sent the campaign",
            "coupons": "distinct coupons in the campaign",
            "redemptions": "coupons redeemed",
            "redeeming_households": "households that redeemed at least one coupon",
            "redemption_rate": "redeeming_households / households_targeted",
        },
    },
    "campaign_exposure": {
        "about": "which household was sent which campaign",
        "columns": {
            "household_id": "household key",
            "campaign_id": "campaign key",
            "campaign_type": "campaign type",
        },
    },
    "department_week": {
        "about": "spend by product department and week",
        "columns": {
            "department": "e.g. 'GROCERY', 'PRODUCE', 'MEAT', 'DRUG GM', 'FUEL'",
            "week": "retailer week 1-53",
            "spend": "USD",
            "baskets": "trips that included the department",
            "households": "households that bought from the department",
        },
    },
    "redemptions": {
        "about": "one row per coupon redemption",
        "columns": {
            "household_id": "household key",
            "campaign_id": "campaign key",
            "week": "retailer week",
            "redemption_date": "date",
        },
    },
    "household_scores": {
        "about": "model outputs as of the last week of data, one row per currently active household",
        "columns": {
            "household_id": "household key",
            "is_regular": "true if the household shopped in >= 8 of the last 16 weeks (only these get a lapse score)",
            "lapse_prob": "probability a regular shopper goes quiet (<= 2 shopping weeks) in the next 8 weeks; NULL if not regular",
            "risk_decile": "1 (lowest risk) to 10 (highest); NULL if not regular",
            "uplift": "estimated extra weekly spend in USD if sent the next coupon booklet",
            "expected_profit": "simulated profit in USD of sending a booklet (28% margin, $1.50 cost)",
            "mail": "true if expected_profit > 0, i.e. the model recommends sending a booklet",
            "spend_16w": "average weekly spend over the last 16 weeks, USD",
            "trips_16w": "trips in the last 16 weeks",
            "recency_weeks": "weeks since the last trip",
        },
    },
}
ALLOWED_TABLES = frozenset(SCHEMA)
DENIED_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "parquet_scan",
        "read_json",
        "read_json_auto",
        "read_json_objects",
        "read_ndjson",
        "read_text",
        "read_blob",
        "glob",
        "sniff_csv",
        "query",
        "query_table",
        "read_xlsx",
        "st_read",
        "postgres_scan",
        "sqlite_scan",
        "mysql_scan",
        "current_setting",
    }
)
FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Copy",
        "Pragma",
        "Set",
        "Use",
        "Attach",
        "Detach",
        "Merge",
        "Transaction",
        "Commit",
        "Rollback",
        "Install",
        "Load",
    )
    if hasattr(exp, name)
)


class UnsafeSQL(ValueError):
    pass


def schema_text() -> str:
    lines = []
    for table, spec in SCHEMA.items():
        lines.append(f"TABLE {table}: {spec['about']}")
        lines += [f"  - {c}: {d}" for c, d in spec["columns"].items()]
    return "\n".join(lines)


SYSTEM_PROMPT = """You write ONE DuckDB SQL query that answers a question about a grocery retailer's data
(2,469 households, one year: 2017, 53 retailer weeks).

Tables you may use (and nothing else):
{schema}

Rules:
- Return only the query inside a ```sql code block. No explanation.
- A single SELECT (CTEs are fine). Never modify data.
- Use only the tables and columns listed above.
- Money is in USD. "Spend" means spend_ex_fuel unless the question mentions fuel.
- Booleans are true/false; cast to INTEGER before averaging them.
- Give readable column aliases and round money to 2 decimals.
"""


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5-coder:7b"
    api_key: str | None = None
    timeout: float = 60.0

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> LLMConfig:
        o = overrides or {}
        return cls(
            base_url=o.get("base_url") or os.environ.get("COUPONROI_LLM_BASE_URL", cls.base_url),
            model=o.get("model") or os.environ.get("COUPONROI_LLM_MODEL", cls.model),
            api_key=o.get("api_key") or os.environ.get("COUPONROI_LLM_API_KEY"),
        )


class LLMClient:
    """Minimal OpenAI-compatible chat client (works with Ollama, Groq, Gemini, OpenAI)."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig.from_env()

    def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        r = httpx.post(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json={"model": self.cfg.model, "messages": messages, "temperature": temperature},
            timeout=self.cfg.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


def guard(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Reject anything that isn't a single read-only SELECT on the whitelisted tables."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeSQL(f"only SELECT queries are allowed, got {type(stmt).__name__}")
    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQL(f"{type(node).__name__} is not allowed")
    for fn in stmt.find_all(exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name in DENIED_FUNCTIONS or name.startswith(("duckdb_", "pragma_")):
            raise UnsafeSQL(f"function {name}() is not allowed")
    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    for t in stmt.find_all(exp.Table):
        if isinstance(t.this, exp.Func) or not t.name:
            raise UnsafeSQL("table functions are not allowed")
        name = t.name.lower()
        if t.db and t.db.lower() not in ("main",):
            raise UnsafeSQL(f"schema {t.db} is not allowed")
        if name not in ALLOWED_TABLES and name not in ctes:
            raise UnsafeSQL(f"table {name} is not allowed")
    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(max_rows)
    else:
        try:
            n = int(limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            n = max_rows + 1
        if n > max_rows:
            stmt = stmt.limit(max_rows)
    return stmt.sql(dialect="duckdb")


def connect_readonly(path: Path = APP_DB) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})


def summary_messages(question: str, sql: str, df: pd.DataFrame) -> list[dict]:
    table = df.head(30).to_csv(index=False)
    return [
        {
            "role": "system",
            "content": "You explain query results to a busy marketing manager in 2-3 plain sentences. "
            "Use ONLY numbers that appear in the result table. If the table can't answer the question, say so. "
            "No preamble.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nSQL used:\n{sql}\n\nResult ({len(df)} rows, first 30):\n{table}",
        },
    ]


@dataclass
class AskResult:
    question: str
    sql: str | None = None
    data: pd.DataFrame | None = None
    summary: str | None = None
    error: str | None = None
    raw: str | None = field(default=None, repr=False)


def ask(
    question: str, con: duckdb.DuckDBPyConnection, client: LLMClient, summarize: bool = True
) -> AskResult:
    res = AskResult(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema_text())},
        {"role": "user", "content": question},
    ]
    try:
        res.raw = client.complete(messages)
        res.sql = guard(extract_sql(res.raw))
        res.data = con.execute(res.sql).df()
        if summarize:
            res.summary = client.complete(summary_messages(question, res.sql, res.data)).strip()
    except UnsafeSQL as e:
        res.error = f"Blocked: {e}"
    except (duckdb.Error, httpx.HTTPError, KeyError) as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


# --------------------------------------------------------------------------- evaluation


def load_eval(path: Path = EVAL_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cell(v):
    """Make values comparable across queries: round numbers, treat True as 1, NaN as None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 2)
    return str(v)


def _normalise(df: pd.DataFrame) -> list[tuple]:
    return sorted((tuple(_cell(v) for v in row) for row in df.itertuples(index=False)), key=repr)


def results_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Execution match: same rows and values, ignoring column names, row order and column order."""
    if a.shape != b.shape:
        return False
    if _normalise(a) == _normalise(b):
        return True

    # same columns in a different order: pair up columns with identical value lists, then compare rows
    def signature(df: pd.DataFrame, i: int) -> tuple:
        return tuple(sorted((_cell(v) for v in df.iloc[:, i]), key=repr))

    remaining = list(range(b.shape[1]))
    order = []
    for i in range(a.shape[1]):
        sig = signature(a, i)
        j = next((j for j in remaining if signature(b, j) == sig), None)
        if j is None:
            return False
        remaining.remove(j)
        order.append(j)
    return _normalise(a) == _normalise(b.iloc[:, order])


def evaluate(con: duckdb.DuckDBPyConnection, client: LLMClient, items: list[dict] | None = None) -> dict:
    items = items or load_eval()
    rows = []
    for it in items:
        ref = con.execute(guard(it["sql"])).df()
        r = ask(it["question"], con, client, summarize=False)
        ok = r.error is None and r.data is not None and results_match(r.data, ref)
        rows.append(
            {"id": it["id"], "question": it["question"], "correct": ok, "error": r.error, "sql": r.sql}
        )
    acc = sum(r["correct"] for r in rows) / len(rows)
    return {"model": client.cfg.model, "execution_accuracy": acc, "n": len(rows), "items": rows}
