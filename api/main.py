"""Scoring API: lapse risk + expected response to a booklet + a mail/skip recommendation.

    uvicorn api.main:app --reload            (from the repo root)
    docker build -t coupon-roi-api . && docker run -p 8000:8000 coupon-roi-api

Endpoints
    GET  /health                 liveness + which models are loaded
    GET  /model-info             training metadata for both models
    GET  /households/{id}        precomputed scores for a known household
    POST /score                  score any feature vector (same features the models were trained on)
"""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("COUPONROI_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

from couponroi.features import FEATURE_DESCRIPTIONS, FEATURES  # noqa: E402
from couponroi.models import PlattCalibrator, shrink  # noqa: E402

MODEL_DIR = ROOT / "artifacts" / "models"
APP_DB = ROOT / "artifacts" / "app" / "app.duckdb"
MIN_REGULAR = 8  # shopped in >= 8 of the last 16 weeks to get a lapse score


class Models:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.lapse = lgb.Booster(model_file=str(model_dir / "lapse_lgbm.txt"))
        self.calibrator = PlattCalibrator.from_dict(
            json.loads((model_dir / "lapse_calibrator.json").read_text())
        )
        self.lapse_meta = json.loads((model_dir / "lapse_meta.json").read_text())
        self.uplift = joblib.load(model_dir / "uplift_model.joblib")
        self.uplift_meta = json.loads((model_dir / "uplift_meta.json").read_text())


@lru_cache(maxsize=1)
def models() -> Models:
    return Models()


class Features(BaseModel):
    """The point-in-time features from sql/features/household_features.sql."""

    recency_weeks: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["recency_weeks"])
    spend_4w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["spend_4w"])
    spend_8w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["spend_8w"])
    spend_16w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["spend_16w"])
    trips_4w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["trips_4w"])
    trips_8w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["trips_8w"])
    trips_16w: float = Field(ge=0, description=FEATURE_DESCRIPTIONS["trips_16w"])
    active_weeks_8w: float = Field(ge=0, le=8, description=FEATURE_DESCRIPTIONS["active_weeks_8w"])
    active_weeks_16w: float = Field(ge=0, le=16, description=FEATURE_DESCRIPTIONS["active_weeks_16w"])
    basket_value_16w: float | None = Field(
        default=None, ge=0, description=FEATURE_DESCRIPTIONS["basket_value_16w"]
    )
    momentum_4v16: float | None = Field(default=None, ge=0, description=FEATURE_DESCRIPTIONS["momentum_4v16"])
    momentum_8v16: float | None = Field(default=None, ge=0, description=FEATURE_DESCRIPTIONS["momentum_8v16"])
    spend_cv_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["spend_cv_16w"])
    spend_slope_rel_16w: float = Field(default=0, description=FEATURE_DESCRIPTIONS["spend_slope_rel_16w"])
    private_share_16w: float = Field(
        default=0, ge=0, le=1, description=FEATURE_DESCRIPTIONS["private_share_16w"]
    )
    promo_share_16w: float = Field(default=0, ge=0, le=1, description=FEATURE_DESCRIPTIONS["promo_share_16w"])
    discount_share_16w: float = Field(
        default=0, ge=0, le=1, description=FEATURE_DESCRIPTIONS["discount_share_16w"]
    )
    buys_fuel_16w: float = Field(default=0, ge=0, le=1, description=FEATURE_DESCRIPTIONS["buys_fuel_16w"])
    redemptions_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["redemptions_16w"])
    departments_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["departments_16w"])
    categories_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["categories_16w"])
    stores_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["stores_16w"])
    campaigns_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["campaigns_16w"])
    type_a_16w: float = Field(default=0, ge=0, description=FEATURE_DESCRIPTIONS["type_a_16w"])


class Business(BaseModel):
    gross_margin: float = Field(default=0.28, gt=0, lt=1)
    cost_per_booklet: float = Field(default=1.50, ge=0)
    campaign_weeks: float = Field(default=7.0, gt=0, le=26)


class ScoreRequest(BaseModel):
    features: Features
    business: Business = Business()

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "recency_weeks": 2,
                    "spend_4w": 18.5,
                    "spend_8w": 31.0,
                    "spend_16w": 42.0,
                    "trips_4w": 2,
                    "trips_8w": 6,
                    "trips_16w": 14,
                    "active_weeks_8w": 4,
                    "active_weeks_16w": 10,
                    "basket_value_16w": 48.0,
                    "momentum_4v16": 0.44,
                    "momentum_8v16": 0.74,
                    "spend_cv_16w": 1.1,
                    "spend_slope_rel_16w": -0.03,
                    "private_share_16w": 0.25,
                    "promo_share_16w": 0.2,
                    "discount_share_16w": 0.15,
                    "buys_fuel_16w": 0,
                    "redemptions_16w": 0,
                    "departments_16w": 9,
                    "categories_16w": 60,
                    "stores_16w": 2,
                    "campaigns_16w": 1,
                    "type_a_16w": 1,
                }
            }
        }
    }


class Driver(BaseModel):
    feature: str
    value: float | None
    effect: float = Field(
        description="contribution to the lapse log-odds (TreeSHAP); positive pushes risk up"
    )


class ScoreResponse(BaseModel):
    is_regular: bool
    lapse_probability: float | None
    lapse_drivers: list[Driver]
    uplift_per_week: float = Field(
        description="expected extra weekly spend (USD) if sent a booklet, after shrinkage"
    )
    expected_profit: float = Field(description="simulated: uplift x weeks x margin - cost")
    recommendation: str
    reason: str


def _usd(x: float) -> str:
    return f"-${abs(x):.2f}" if x < 0 else f"${x:.2f}"


def score_frame(X: pd.DataFrame, business: Business) -> list[ScoreResponse]:
    m = models()
    X = X[FEATURES].astype(float)
    raw = m.lapse.predict(X)
    contrib = m.lapse.predict(X, pred_contrib=True)[:, :-1]
    up_meta = m.uplift_meta
    uplift = shrink(m.uplift.predict(X), up_meta["ate"], up_meta["shrinkage"], up_meta["center"])
    out = []
    for i in range(len(X)):
        regular = bool(X.iloc[i]["active_weeks_16w"] >= MIN_REGULAR)
        order = np.argsort(-np.abs(contrib[i]))[:3]
        drivers = (
            [
                Driver(
                    feature=FEATURE_DESCRIPTIONS[FEATURES[j]],
                    value=None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j]),
                    effect=round(float(contrib[i, j]), 4),
                )
                for j in order
            ]
            if regular
            else []
        )
        profit = float(
            uplift[i] * business.campaign_weeks * business.gross_margin - business.cost_per_booklet
        )
        breakeven = business.cost_per_booklet / (business.campaign_weeks * business.gross_margin)
        out.append(
            ScoreResponse(
                is_regular=regular,
                lapse_probability=float(m.calibrator(np.array([raw[i]]))[0]) if regular else None,
                lapse_drivers=drivers,
                uplift_per_week=round(float(uplift[i]), 4),
                expected_profit=round(profit, 4),
                recommendation="mail" if profit > 0 else "skip",
                reason=(
                    f"expected response {_usd(uplift[i])}/wk vs break-even {_usd(breakeven)}/wk "
                    f"({business.gross_margin:.0%} margin, ${business.cost_per_booklet:.2f} per booklet, "
                    f"{business.campaign_weeks:g} weeks)"
                ),
            )
        )
    return out


app = FastAPI(
    title="Coupon ROI & Targeting API",
    version="1.0.0",
    description="Lapse risk, booklet uplift and a mail/skip decision per household. Money figures are simulated.",
)


@app.get("/health")
def health() -> dict:
    try:
        m = models()
        return {
            "status": "ok",
            "lapse_model_trees": m.lapse.num_trees(),
            "uplift_learner": m.uplift_meta["learner"],
        }
    except FileNotFoundError as e:
        raise HTTPException(503, f"model files missing: {e}. Run `couponroi all` first.") from e


@app.get("/model-info")
def model_info() -> dict:
    m = models()
    return {
        "lapse": m.lapse_meta,
        "uplift": {k: v for k, v in m.uplift_meta.items() if k != "features"},
        "features": FEATURES,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    X = pd.DataFrame([req.features.model_dump()])
    return score_frame(X, req.business)[0]


@app.get("/households/{household_id}", response_model=ScoreResponse)
def household(
    household_id: int, margin: float = 0.28, cost: float = 1.50, weeks: float = 7.0
) -> ScoreResponse:
    if not APP_DB.exists():
        raise HTTPException(503, "app bundle missing. Run `couponroi export`.")
    con = duckdb.connect(str(APP_DB), read_only=True)
    try:
        df = con.execute(
            f"SELECT {', '.join(FEATURES)} FROM household_scores WHERE household_id = ?", [household_id]
        ).df()
    finally:
        con.close()
    if df.empty:
        raise HTTPException(404, f"household {household_id} is not active in the last 8 weeks of data")
    return score_frame(df, Business(gross_margin=margin, cost_per_booklet=cost, campaign_weeks=weeks))[0]
