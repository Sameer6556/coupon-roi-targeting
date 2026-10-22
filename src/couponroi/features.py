"""Point-in-time household features and lapse labels.

The same feature function feeds three things: the lapse-risk model, the uplift model
(features as of the week before a campaign wave), and the API. One definition of
"what we knew about a household at week X" means no train/serve skew.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from . import config

FEATURE_SQL = config.SQL_DIR / "features" / "household_features.sql"
LABEL_SQL = config.SQL_DIR / "features" / "lapse_label.sql"

FEATURES = [
    "recency_weeks",
    "spend_4w",
    "spend_8w",
    "spend_16w",
    "trips_4w",
    "trips_8w",
    "trips_16w",
    "active_weeks_8w",
    "active_weeks_16w",
    "basket_value_16w",
    "momentum_4v16",
    "momentum_8v16",
    "spend_cv_16w",
    "spend_slope_rel_16w",
    "private_share_16w",
    "promo_share_16w",
    "discount_share_16w",
    "buys_fuel_16w",
    "redemptions_16w",
    "departments_16w",
    "categories_16w",
    "stores_16w",
    "campaigns_16w",
    "type_a_16w",
]

FEATURE_DESCRIPTIONS = {
    "recency_weeks": "weeks since last shopping trip",
    "spend_4w": "avg weekly spend, last 4 weeks ($)",
    "spend_8w": "avg weekly spend, last 8 weeks ($)",
    "spend_16w": "avg weekly spend, last 16 weeks ($)",
    "trips_4w": "trips in the last 4 weeks",
    "trips_8w": "trips in the last 8 weeks",
    "trips_16w": "trips in the last 16 weeks",
    "active_weeks_8w": "weeks with a trip, last 8",
    "active_weeks_16w": "weeks with a trip, last 16",
    "basket_value_16w": "avg spend per trip ($)",
    "momentum_4v16": "last-4-week spend / last-16-week spend",
    "momentum_8v16": "last-8-week spend / last-16-week spend",
    "spend_cv_16w": "week-to-week spend volatility (CV)",
    "spend_slope_rel_16w": "16-week spend trend, % of avg per week",
    "private_share_16w": "share of spend on store brands",
    "promo_share_16w": "share of spend on displayed/mailer items",
    "discount_share_16w": "discounts / (spend + discounts)",
    "buys_fuel_16w": "bought fuel in the last 16 weeks",
    "redemptions_16w": "coupons redeemed, last 16 weeks",
    "departments_16w": "departments shopped, last 16 weeks",
    "categories_16w": "product categories bought, last 16 weeks",
    "stores_16w": "distinct stores visited, last 16 weeks",
    "campaigns_16w": "campaigns received, last 16 weeks",
    "type_a_16w": "Type A booklets received, last 16 weeks",
}


def features_at(con: duckdb.DuckDBPyConnection, cutoff: int, lookback: int | None = None) -> pd.DataFrame:
    lookback = lookback or config.SETTINGS.churn.lookback
    df = con.execute(FEATURE_SQL.read_text(), {"cutoff": int(cutoff), "lookback": int(lookback)}).df()
    df[FEATURES] = df[FEATURES].astype(float)
    return df


def labels_at(
    con: duckdb.DuckDBPyConnection,
    cutoff: int,
    horizon: int | None = None,
    lookback: int | None = None,
    drop: float | None = None,
) -> pd.DataFrame:
    c = config.SETTINGS.churn
    params = {
        "cutoff": int(cutoff),
        "horizon": int(horizon or c.horizon),
        "lookback": int(lookback or c.lookback),
        "drop": float(drop if drop is not None else c.drop_threshold),
        "min_regular": int(c.min_regular),
        "quiet_max": int(c.quiet_max),
    }
    return con.execute(LABEL_SQL.read_text(), params).df()


def is_regular(df: pd.DataFrame) -> pd.Series:
    """The scoring population: shopped in at least `min_regular` of the last 16 weeks."""
    return df["active_weeks_16w"] >= config.SETTINGS.churn.min_regular


def training_frame(con: duckdb.DuckDBPyConnection, cutoffs: list[int]) -> pd.DataFrame:
    """Stack (features, label) rows for many cutoffs: one row per regular household per cutoff."""
    frames = []
    for c in cutoffs:
        f = features_at(con, c)
        y = labels_at(con, c)[["household_id", "lapse", "spend_halved", "full_lapse", "future_weekly"]]
        frames.append(f.merge(y, on="household_id", how="inner"))
    return pd.concat(frames, ignore_index=True)
