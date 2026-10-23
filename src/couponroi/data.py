"""Download the raw tables and check them before anything else touches them.

Source: the `completejourney_py` package, which ships the 84.51 Complete Journey 2.0
data (2,469 households, one year of grocery transactions) from the CC0-licensed
`completejourney` R package. Using a pip package instead of a manual download means
anyone can rebuild the project with one command, and no data lives in this repo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

# column -> kind ("int", "float", "str", "datetime")
SCHEMAS: dict[str, dict[str, str]] = {
    "transactions": {
        "household_id": "int",
        "store_id": "int",
        "basket_id": "int",
        "product_id": "int",
        "quantity": "int",
        "sales_value": "float",
        "retail_disc": "float",
        "coupon_disc": "float",
        "coupon_match_disc": "float",
        "week": "int",
        "transaction_timestamp": "datetime",
    },
    "products": {
        "product_id": "int",
        "manufacturer_id": "int",
        "department": "str",
        "brand": "str",
        "product_category": "str",
        "product_type": "str",
        "package_size": "str",
    },
    "demographics": {
        "household_id": "int",
        "age": "str",
        "income": "str",
        "home_ownership": "str",
        "marital_status": "str",
        "household_size": "str",
        "household_comp": "str",
        "kids_count": "str",
    },
    "campaigns": {"campaign_id": "int", "household_id": "int"},
    "campaign_descriptions": {
        "campaign_id": "int",
        "campaign_type": "str",
        "start_date": "datetime",
        "end_date": "datetime",
    },
    "coupons": {"coupon_upc": "int", "product_id": "int", "campaign_id": "int"},
    "coupon_redemptions": {
        "household_id": "int",
        "coupon_upc": "int",
        "campaign_id": "int",
        "redemption_date": "datetime",
    },
    "promotions": {
        "product_id": "int",
        "store_id": "int",
        "display_location": "str",
        "mailer_location": "str",
        "week": "int",
    },
}

UNIQUE_KEYS = {
    "products": ["product_id"],
    "demographics": ["household_id"],
    "campaign_descriptions": ["campaign_id"],
    "campaigns": ["campaign_id", "household_id"],
}


@dataclass
class ValidationIssue:
    table: str
    check: str
    detail: str
    severity: str  # "error" stops the pipeline, "warning" is logged and recorded


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "float":
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
    if kind == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    # strings may come back as object, str or category depending on the pandas version
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def validate(tables: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    """Schema, key and sanity checks. Cheap to run, saves hours of confused debugging later."""
    issues: list[ValidationIssue] = []
    for name, schema in SCHEMAS.items():
        if name not in tables:
            issues.append(ValidationIssue(name, "present", "table missing", "error"))
            continue
        df = tables[name]
        missing = [c for c in schema if c not in df.columns]
        if missing:
            issues.append(ValidationIssue(name, "columns", f"missing {missing}", "error"))
            continue
        for col, kind in schema.items():
            if not _kind_ok(df[col], kind):
                issues.append(
                    ValidationIssue(name, "dtype", f"{col} is {df[col].dtype}, expected {kind}", "error")
                )
        if len(df) == 0:
            issues.append(ValidationIssue(name, "rows", "table is empty", "error"))
        keys = UNIQUE_KEYS.get(name)
        if keys and df.duplicated(keys).any():
            n = int(df.duplicated(keys).sum())
            issues.append(ValidationIssue(name, "unique", f"{n} duplicate rows on {keys}", "error"))

    tx = tables.get("transactions")
    if tx is not None and "sales_value" in tx:
        neg = int((tx["sales_value"] < 0).sum())
        if neg:
            issues.append(
                ValidationIssue("transactions", "sales_value>=0", f"{neg} negative rows", "warning")
            )
        zero_qty = int((tx["quantity"] == 0).sum())
        if zero_qty:
            issues.append(
                ValidationIssue(
                    "transactions", "quantity>0", f"{zero_qty} rows with quantity 0 (kept)", "warning"
                )
            )
    prods = tables.get("products")
    if tx is not None and prods is not None:
        unknown = int((~tx["product_id"].isin(prods["product_id"])).sum())
        if unknown:
            issues.append(
                ValidationIssue(
                    "transactions",
                    "product fk",
                    f"{unknown} rows reference products missing from the product table (mapped to UNKNOWN)",
                    "warning",
                )
            )
    camps, desc = tables.get("campaigns"), tables.get("campaign_descriptions")
    if camps is not None and desc is not None:
        orphan = set(camps["campaign_id"]) - set(desc["campaign_id"])
        if orphan:
            issues.append(
                ValidationIssue("campaigns", "campaign fk", f"unknown campaigns {sorted(orphan)}", "error")
            )
    if desc is not None and (desc["end_date"] < desc["start_date"]).any():
        issues.append(
            ValidationIssue("campaign_descriptions", "dates", "end_date before start_date", "error")
        )
    return issues


def load_source(synthetic: bool = False, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    if synthetic:
        from .synthetic import make_tables

        return make_tables(seed=seed)
    from completejourney_py import get_data

    return get_data(list(config.RAW_TABLES))


def fetch(raw_dir: Path = config.RAW_DIR, synthetic: bool = False, seed: int = config.SEED) -> dict:
    """Load every raw table, validate it, and write it to `raw_dir` as parquet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables = load_source(synthetic=synthetic, seed=seed)
    issues = validate(tables)
    for issue in issues:
        (log.error if issue.severity == "error" else log.warning)(
            "%s | %s | %s", issue.table, issue.check, issue.detail
        )
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValueError(f"raw data failed validation: {[(i.table, i.check, i.detail) for i in errors]}")

    manifest = {"source": "synthetic" if synthetic else "completejourney_py", "tables": {}, "issues": []}
    for name, df in tables.items():
        path = raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": int(len(df)), "columns": list(df.columns)}
    manifest["issues"] = [i.__dict__ for i in issues]
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d raw tables to %s", len(tables), raw_dir)
    return manifest
