"""Build the slim bundle the Streamlit app (and the API's household lookup) runs on.

artifacts/app/
    app.duckdb     the semantic tables + model scores (a few MB, safe to commit)
    *.json         results from every analysis step

The app never needs the raw data or the training stack, which keeps deployment to
"pip install -r requirements.txt && streamlit run app/Home.py".
"""

from __future__ import annotations

import json
import logging
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_DESCRIPTIONS, FEATURES
from .warehouse import connect

log = logging.getLogger(__name__)
TABLES = {
    "household_week": "v_household_week",
    "households": "v_households",
    "campaigns": "v_campaigns",
    "campaign_exposure": "v_campaign_exposure",
    "department_week": "v_department_week",
    "redemptions": "v_redemptions",
}
RESULTS = ["data_quality", "eda", "causal", "churn", "uplift", "policy", "drift"]


def lapse_drivers(scores: pd.DataFrame) -> pd.Series:
    """Top reasons behind each regular shopper's lapse score, from LightGBM's built-in
    TreeSHAP (pred_contrib=True), so the app doesn't need the shap package."""
    booster = lgb.Booster(model_file=str(config.MODEL_DIR / "lapse_lgbm.txt"))
    reg = scores["is_regular"].fillna(False).astype(bool)
    out = pd.Series([None] * len(scores), index=scores.index, dtype=object)
    if not reg.any():
        return out
    X = scores.loc[reg, FEATURES].astype(float)
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # last column is the bias term
    for row_idx, (i, row) in enumerate(X.iterrows()):
        c = contrib[row_idx]
        order = np.argsort(-np.abs(c))[:4]
        out.at[i] = json.dumps(
            [
                {
                    "feature": FEATURE_DESCRIPTIONS[FEATURES[j]],
                    "value": round(float(row.iloc[j]), 3),
                    "effect": round(float(c[j]), 3),
                }
                for j in order
            ]
        )
    return out


def run() -> dict:
    config.APP_DIR.mkdir(parents=True, exist_ok=True)
    db_path = config.APP_DIR / "app.duckdb"
    if db_path.exists():
        db_path.unlink()

    wh = connect()
    frames = {name: wh.execute(f"SELECT * FROM {view}").df() for name, view in TABLES.items()}
    wh.close()

    scored = pd.read_parquet(config.ARTIFACTS_DIR / "policy_scored.parquet")
    lapse = pd.read_parquet(config.ARTIFACTS_DIR / "lapse_scores_latest.parquet")[
        ["household_id", "risk_decile"]
    ]
    scored = scored.merge(lapse, on="household_id", how="left")
    scored["top_drivers"] = lapse_drivers(scored)
    scored["mail"] = scored["expected_profit"] > 0
    keep = [
        "household_id",
        "is_regular",
        "lapse_prob",
        "risk_decile",
        "uplift",
        "uplift_raw",
        "expected_profit",
        "mail",
        "top_drivers",
    ] + FEATURES
    frames["household_scores"] = scored[keep]

    con = connect(db_path, read_only=False)
    for name, df in frames.items():
        con.register("df_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")
    con.execute("CHECKPOINT")
    counts = {name: int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]) for name in frames}
    con.close()

    for r in RESULTS:
        src = config.RESULTS_DIR / f"{r}.json"
        if src.exists():
            shutil.copy(src, config.APP_DIR / f"{r}.json")
    for m in ("lapse_meta.json", "uplift_meta.json"):
        if (config.MODEL_DIR / m).exists():
            shutil.copy(config.MODEL_DIR / m, config.APP_DIR / m)

    size_mb = db_path.stat().st_size / 1e6
    log.info("app bundle: %s rows, app.duckdb %.1f MB", counts, size_mb)
    return {"tables": counts, "size_mb": size_mb}
