"""Every path, window and business assumption in one place.

If an interviewer asks "where did that number come from", the answer should be
in this file. Business assumptions (margin, costs) are deliberately explicit
because the data has no margins or costs: every money figure in this project
is simulated from these values and the sensitivity analysis varies them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("COUPONROI_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "sql").is_dir():
            return parent
    return Path.cwd()


ROOT = _find_root()
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
APP_DIR = ARTIFACTS_DIR / "app"
MLFLOW_DIR = ROOT / "mlruns"
EVAL_DIR = ROOT / "eval"

SEED = 42

RAW_TABLES = (
    "transactions",
    "products",
    "demographics",
    "campaigns",
    "campaign_descriptions",
    "coupons",
    "coupon_redemptions",
    "promotions",
)


@dataclass(frozen=True)
class CausalConfig:
    """Settings for the campaign impact analysis (matched difference-in-differences)."""

    focal_type: str = "Type A"  # the big personalised coupon booklets (~1,100 households each)
    pre_weeks: int = 8  # baseline window before a campaign starts
    min_pre_weeks: int = 4  # skip a wave if the clean baseline would be shorter than this
    first_usable_week: int = 2  # week 1 is a single Sunday, so it is not a full week
    # max distance on the logit propensity, in SDs. 0.2 is the textbook default (Austin 2011);
    # 0.05 gave the best balance here (mean |SMD| 0.04) for a modest loss of pairs.
    # See the robustness table for 0.2 and for matching with replacement.
    caliper_sd: float = 0.05
    # matched exactly on these (then on propensity inside each cell): whether the household
    # got another campaign during the wave, and how many earlier booklets it had (0/1/2+)
    exact_on: tuple[str, ...] = ("co_exposure_post_any", "prior_waves_bucket")
    # inside the caliper, pick the closest on these (Mahalanobis distance)
    mahalanobis_on: tuple[str, ...] = ("pre_spend", "prior_spend", "pre_active_share", "pre_trips")
    with_replacement: bool = False  # tried True: matches 98% of recipients but reuses controls 3-4x (noisier)
    bootstrap_reps: int = 200  # for the weighting-based DR-DiD check
    outcome: str = "spend_xf"  # weekly spend excluding fuel
    covariates: tuple[str, ...] = (
        "pre_spend",
        "pre_trips",
        "pre_active_share",
        "pre_departments",
        "pre_private_share",
        "pre_promo_share",
        "pre_coupon_use",
        "pre_trend",
        "prior_spend",
        "prior_waves",
        "co_exposure_pre",
        "co_exposure_post",
    )


@dataclass(frozen=True)
class ChurnConfig:
    """Settings for the lapse-risk model.

    Population: regular shoppers (shopped in at least `min_regular` of the last
    `lookback` weeks). Target: they shop in at most `quiet_max` of the next `horizon`
    weeks. A regular going quiet is a real change in behaviour. (Tried first: "spend
    halves" over all active households. It is mostly noise, AUC ~0.70; see docs.)
    """

    lookback: int = 16
    horizon: int = 8
    min_regular: int = 8  # shopped in >= 8 of the last 16 weeks
    quiet_max: int = 2  # ...and then in <= 2 of the next 8 weeks
    drop_threshold: float = 0.5  # secondary label: spend over the next 8 weeks < 50% of baseline
    first_cutoff: int = 17  # need `lookback` weeks of history (weeks 2..17)
    last_label_week: int = 53
    cutoff_step: int = 2
    test_cutoffs: tuple[int, ...] = (37, 41, 45)  # rolling-origin backtest folds
    calibration_cutoffs: int = 2  # latest N labelled cutoffs before a test cutoff are used for calibration
    top_k_share: float = 0.10  # "who would we contact" budget used for recall@k

    def cutoffs(self) -> list[int]:
        last = self.last_label_week - self.horizon
        return list(range(self.first_cutoff, last + 1, self.cutoff_step))


@dataclass(frozen=True)
class BusinessConfig:
    """Assumptions for the targeting simulation. None of these are in the data."""

    gross_margin: float = 0.28  # grocery gross margins sit around 25-30%
    cost_per_contact: float = 1.50  # print + mail one coupon booklet, USD
    margin_grid: tuple[float, ...] = (0.20, 0.25, 0.28, 0.30, 0.35)
    cost_grid: tuple[float, ...] = (0.50, 1.00, 1.50, 2.00, 3.00)


@dataclass(frozen=True)
class Settings:
    causal: CausalConfig = field(default_factory=CausalConfig)
    churn: ChurnConfig = field(default_factory=ChurnConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)


SETTINGS = Settings()


def ensure_dirs() -> None:
    for p in (RAW_DIR, FIG_DIR, RESULTS_DIR, MODEL_DIR, APP_DIR):
        p.mkdir(parents=True, exist_ok=True)
