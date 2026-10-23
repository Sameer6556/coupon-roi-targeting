"""Did the Type A coupon booklets actually change what households spend?

Recipients were picked from their purchase history, so they already spent ~2.5x more
than non-recipients before any booklet arrived. Comparing the two groups afterwards
mostly measures the targeting rule. The design here:

1. Baseline window before each wave (trimmed so it never overlaps the previous wave).
2. Propensity-score matching: each recipient is paired with a non-recipient who looked
   the same beforehand (spend, trips, trend, store-brand and promo habits, coupon use,
   earlier booklets, other campaigns received at the same time).
3. Difference-in-differences on the matched pairs: (change in weekly spend for the
   recipient) minus (change for its match). Shared shocks like holidays cancel out.
4. Checks: covariate balance, an event study for pre-trends, a placebo with a fake
   start date inside the baseline, and a set of robustness specs.
5. CUPED and a power analysis to size a proper randomised holdout for next time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression

from . import config
from . import plotting as P
from .warehouse import connect

log = logging.getLogger(__name__)
Z95 = float(stats.norm.ppf(0.975))
WAVE_SQL = config.SQL_DIR / "features" / "wave_units.sql"
SKEWED = {
    "pre_spend",
    "pre_trips",
    "prior_spend",
    "pre_departments",
    "prior_waves",
    "co_exposure_pre",
    "co_exposure_post",
}


# --------------------------------------------------------------------------- waves


@dataclass(frozen=True)
class Wave:
    wave: int
    campaign_id: int
    pre_start: int
    start: int
    end: int

    @property
    def pre_weeks(self) -> int:
        return self.start - self.pre_start

    @property
    def post_weeks(self) -> int:
        return self.end - self.start + 1

    @property
    def label(self) -> str:
        return f"wave {self.wave} (campaign {self.campaign_id})"


def list_waves(
    con: duckdb.DuckDBPyConnection, cfg: config.CausalConfig = config.SETTINGS.causal
) -> list[Wave]:
    camps = con.execute(
        "SELECT campaign_id, start_week, end_week FROM dim_campaign WHERE campaign_type = ? ORDER BY start_week",
        [cfg.focal_type],
    ).df()
    max_week = int(con.execute("SELECT max(week) FROM dim_week").fetchone()[0])
    waves: list[Wave] = []
    prev_end: int | None = None
    for row in camps.itertuples():
        start, end = int(row.start_week), min(int(row.end_week), max_week)
        pre_start = max(cfg.first_usable_week, start - cfg.pre_weeks)
        if prev_end is not None:
            pre_start = max(pre_start, prev_end + 1)  # never let the baseline overlap the last wave
        prev_end = int(row.end_week)
        if start > max_week or start - pre_start < cfg.min_pre_weeks:
            log.info("skipping campaign %s: baseline would be %d weeks", row.campaign_id, start - pre_start)
            continue
        waves.append(Wave(len(waves) + 1, int(row.campaign_id), pre_start, start, end))
    return waves


def unit_table(
    con: duckdb.DuckDBPyConnection, w: Wave, cfg: config.CausalConfig = config.SETTINGS.causal
) -> pd.DataFrame:
    params = {
        "campaign_id": w.campaign_id,
        "pre_start": w.pre_start,
        "start": w.start,
        "end": w.end,
        "first_week": cfg.first_usable_week,
    }
    df = con.execute(WAVE_SQL.read_text(), params).df()
    df["wave"] = w.wave
    df["campaign_id"] = w.campaign_id
    return df


# --------------------------------------------------------------------------- matching


def design_matrix(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """log1p the skewed counts/amounts, standardise, drop columns with no variation."""
    X = df[covariates].astype(float).copy()
    for c in covariates:
        if c in SKEWED:
            X[c] = np.log1p(X[c].clip(lower=0))
    sd = X.std(ddof=0)
    keep = sd[sd > 0].index
    return (X[keep] - X[keep].mean()) / sd[keep]


def propensity(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    Z = design_matrix(df, covariates)
    model = LogisticRegression(max_iter=5000, C=1.0)
    model.fit(Z.to_numpy(), df["treated"].to_numpy())
    return np.clip(model.predict_proba(Z.to_numpy())[:, 1], 1e-6, 1 - 1e-6)


def greedy_match(
    treated: np.ndarray,
    logit_ps: np.ndarray,
    caliper_sd: float,
    strata: np.ndarray | None = None,
    key: np.ndarray | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    """1:1 matching without replacement: Mahalanobis distance on a few key covariates,
    but only among non-recipients inside the propensity caliper (Rubin & Thomas 2000).

    * The propensity caliper makes sure the match looks similar on everything.
    * The Mahalanobis step picks, among those, the closest on what matters most for the
      outcome (spend level and activity), which is what drives balance on spend.
    * `strata` forces exact matches on a few discrete variables.
    * Recipients with the highest propensity go first: they are the hardest to match.

    With `replace=True` a non-recipient can be the match for several recipients. That
    stops the few heavy-spending non-recipients from running out (better balance) at
    the cost of a smaller effective sample; SEs are clustered by household to match.

    Recipients with no candidate inside the caliper are dropped, so the estimate is for
    recipients that HAVE a comparable non-recipient. The report says how many that is.
    """
    caliper = caliper_sd * float(np.std(logit_ps))
    strata = np.zeros(len(treated), dtype=int) if strata is None else np.asarray(strata)
    t_idx = np.flatnonzero(treated == 1)
    c_idx = np.flatnonzero(treated == 0)
    order = t_idx[np.argsort(-logit_ps[t_idx], kind="stable")]
    lc, sc = logit_ps[c_idx], strata[c_idx]
    if key is not None:
        inv = np.linalg.pinv(np.cov(key, rowvar=False))
        kc = key[c_idx]
    available = np.ones(len(c_idx), dtype=bool)
    pairs = []
    for i in order:
        gap = np.abs(lc - logit_ps[i])
        ok = available & (sc == strata[i]) & (gap <= caliper)
        if not ok.any():
            continue
        cand = np.flatnonzero(ok)
        if key is None:
            j = cand[np.argmin(gap[cand])]
        else:
            diff = kc[cand] - key[i]
            j = cand[np.argmin(np.einsum("ij,jk,ik->i", diff, inv, diff))]
        if not replace:
            available[j] = False
        pairs.append((i, c_idx[j], float(gap[j])))
    return pd.DataFrame(pairs, columns=["t", "c", "distance"])


def add_strata_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co_exposure_post_any"] = (df["co_exposure_post"] > 0).astype(int)
    df["prior_waves_bucket"] = df["prior_waves"].clip(upper=2).astype(int)
    return df


def smd(df: pd.DataFrame, pairs: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Standardised mean differences before and after matching.

    Both use the pre-matching pooled SD as the denominator, so a smaller "after" value
    means the means really got closer, not that the variance shrank. |SMD| < 0.1 is
    the usual bar for good balance.
    """
    t, c = df[df.treated == 1], df[df.treated == 0]
    mt, mc = df.iloc[pairs.t], df.iloc[pairs.c]
    rows = []
    for cov in covariates:
        denom = np.sqrt((t[cov].var() + c[cov].var()) / 2) or 1.0
        rows.append(
            {
                "covariate": cov,
                "before": float((t[cov].mean() - c[cov].mean()) / denom),
                "after": float((mt[cov].mean() - mc[cov].mean()) / denom),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- estimators


def summarize(est: float, se: float) -> dict:
    est, se = float(est), float(se)
    p = float(2 * (1 - stats.norm.cdf(abs(est / se)))) if se > 0 else float("nan")
    return {"est": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se, "p": p}


def welch(a: pd.Series, b: pd.Series) -> dict:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return summarize(a.mean() - b.mean(), se)


def paired(d: np.ndarray) -> dict:
    return summarize(d.mean(), d.std(ddof=1) / np.sqrt(len(d)))


def ols_effect(y: pd.Series, X: pd.DataFrame, groups: pd.Series | None = None) -> dict:
    X = sm.add_constant(X.astype(float), has_constant="add")
    if groups is None:
        fit = sm.OLS(y.astype(float), X).fit(cov_type="HC1")
    else:
        fit = sm.OLS(y.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return summarize(fit.params["treated"], fit.bse["treated"])


def stacked_effect(m: pd.DataFrame, y: str = "dy") -> dict:
    """Pooled effect across waves: y ~ treated + wave dummies, SEs clustered by household
    (the same household can show up in several waves)."""
    X = pd.get_dummies(m["wave"].astype(str), prefix="wave", drop_first=True, dtype=float)
    X.insert(0, "treated", m["treated"].astype(float).to_numpy())
    X.index = m.index
    return ols_effect(m[y], X, groups=m["household_id"])


def dr_did_wave(df: pd.DataFrame, covariates: list[str], trim: float = 0.99) -> tuple[float, int]:
    """Doubly robust DiD for two periods (Sant'Anna & Zhao 2020), no matching at all.

    ATT = mean over recipients of (dY - m0(X)) - weighted mean over non-recipients of
    (dY - m0(X)), where m0 is a regression of dY on X fitted on non-recipients and the
    weights are p(X)/(1-p(X)). It is consistent if EITHER the propensity model OR the
    outcome model is right, which is why it's a good independent check on matching.
    Units with propensity above `trim` have no real comparison and are dropped.
    """
    ps = propensity(df, covariates)
    keep = ps <= trim
    d, ps = df[keep].reset_index(drop=True), ps[keep]
    X = sm.add_constant(design_matrix(d, covariates).to_numpy(), has_constant="add")
    D = d["treated"].to_numpy()
    dy = d["dy"].to_numpy()
    beta, *_ = np.linalg.lstsq(X[D == 0], dy[D == 0], rcond=None)
    resid = dy - X @ beta
    w0 = ps * (1 - D) / (1 - ps)
    att = resid[D == 1].mean() - np.sum(w0 * resid) / np.sum(w0)
    return float(att), int((~keep).sum())


def dr_did(
    con: duckdb.DuckDBPyConnection, waves: list[Wave], cfg: config.CausalConfig, seed: int = config.SEED
) -> dict:
    """Pooled DR-DiD (treated-weighted average of waves) with a household cluster bootstrap."""
    covs = list(cfg.covariates)
    units = {w.wave: unit_table(con, w, cfg) for w in waves}

    def pooled(frames: dict[int, pd.DataFrame]) -> float:
        atts, weights = [], []
        for df in frames.values():
            att, _ = dr_did_wave(df, covs)
            atts.append(att)
            weights.append(df.treated.sum())
        return float(np.average(atts, weights=weights))

    est = pooled(units)
    trimmed = sum(dr_did_wave(df, covs)[1] for df in units.values())
    rng = np.random.default_rng(seed)
    households = np.unique(np.concatenate([df.household_id.to_numpy() for df in units.values()]))
    draws = []
    for _ in range(cfg.bootstrap_reps):
        sample = pd.Series(rng.choice(households, size=len(households), replace=True)).value_counts()
        frames = {}
        for k, df in units.items():
            reps = df.household_id.map(sample).fillna(0).astype(int).to_numpy()
            frames[k] = df.loc[df.index.repeat(reps)].reset_index(drop=True)
        draws.append(pooled(frames))
    se = float(np.std(draws, ddof=1))
    return {**summarize(est, se), "trimmed_units": int(trimmed), "bootstrap_reps": cfg.bootstrap_reps}


# --------------------------------------------------------------------------- one wave


def analyze_wave(
    con: duckdb.DuckDBPyConnection,
    w: Wave,
    cfg: config.CausalConfig = config.SETTINGS.causal,
    covariates: list[str] | None = None,
    caliper_sd: float | None = None,
    subset: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    exact_on: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    covariates = list(covariates or cfg.covariates)
    caliper_sd = cfg.caliper_sd if caliper_sd is None else caliper_sd
    exact_on = list(cfg.exact_on if exact_on is None else exact_on)
    df = add_strata_columns(unit_table(con, w, cfg))
    if subset is not None:
        df = subset(df).reset_index(drop=True)
    ps = propensity(df, covariates)
    logit = np.log(ps / (1 - ps))
    df["ps"] = ps
    strata = df[exact_on].astype(str).agg("|".join, axis=1).to_numpy() if exact_on else None
    key_cols = [c for c in cfg.mahalanobis_on if c in covariates]
    key = design_matrix(df, key_cols).to_numpy() if key_cols else None
    pairs = greedy_match(
        df["treated"].to_numpy(), logit, caliper_sd, strata=strata, key=key, replace=cfg.with_replacement
    )
    pairs["pair_id"] = [f"{w.wave}-{i}" for i in range(len(pairs))]

    mt = df.iloc[pairs.t].assign(pair_id=pairs.pair_id.to_numpy())
    mc = df.iloc[pairs.c].assign(pair_id=pairs.pair_id.to_numpy())
    matched = pd.concat([mt, mc], ignore_index=True)

    t_all, c_all = df[df.treated == 1], df[df.treated == 0]
    baseline = float(mt.pre_spend.mean())
    # treated - control difference in spend change, SE clustered by household because a
    # non-recipient can be reused as the match for several recipients
    did = ols_effect(matched.dy, matched[["treated"]], groups=matched.household_id)
    Z = design_matrix(matched, covariates)
    Z.insert(0, "treated", matched.treated.astype(float))
    dr = ols_effect(matched.dy, Z, groups=matched.household_id)
    bal = smd(df, pairs, covariates)

    res = {
        "wave": w.wave,
        "campaign_id": w.campaign_id,
        "pre_window": [w.pre_start, w.start - 1],
        "post_window": [w.start, w.end],
        "n_treated": int(len(t_all)),
        "n_control": int(len(c_all)),
        "n_pairs": int(len(pairs)),
        "share_treated_matched": float(len(pairs) / max(len(t_all), 1)),
        "baseline_weekly_spend": baseline,
        "treated_pre_spend_all": float(t_all.pre_spend.mean()),
        "control_pre_spend_all": float(c_all.pre_spend.mean()),
        "naive_post": welch(t_all.post_spend, c_all.post_spend),
        "did_unmatched": welch(t_all.dy, c_all.dy),
        "did_matched": did,
        "did_matched_dr": dr,
        "did_matched_trips": ols_effect(matched.dy_trips, matched[["treated"]], groups=matched.household_id),
        "unique_controls": int(mc.household_id.nunique()),
        "rel_lift": did["est"] / baseline if baseline else float("nan"),
        "balance": bal.to_dict("records"),
        "max_abs_smd_before": float(bal.before.abs().max()),
        "max_abs_smd_after": float(bal.after.abs().max()),
    }
    return res, matched


# --------------------------------------------------------------------------- pooled pieces


def event_study(con: duckdb.DuckDBPyConnection, waves: list[Wave], matched: pd.DataFrame) -> pd.DataFrame:
    """Weekly treated-minus-control gap, relative to each household's own baseline average.

    Flat points before week 0 are the visual check of parallel pre-trends.
    """
    frames = []
    for w in waves:
        m = matched[matched.wave == w.wave][["household_id", "treated"]]
        s = con.execute(
            "SELECT household_id, week, spend_xf FROM hh_week WHERE week BETWEEN ? AND ?",
            [w.pre_start, w.end],
        ).df()
        s = s.merge(m, on="household_id", how="inner")
        base = s[s.week < w.start].groupby("household_id").spend_xf.mean()
        s["y"] = s.spend_xf - s.household_id.map(base)
        s["rel"] = s.week - w.start
        s["wave"] = w.wave
        frames.append(s)
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for rel, g in long.groupby("rel"):
        if g.wave.nunique() == 1:
            eff = ols_effect(g.y, g[["treated"]], groups=g.household_id)
        else:
            eff = stacked_effect(g.rename(columns={"y": "dy"}), y="dy")
        rows.append({"rel_week": int(rel), "waves": int(g.wave.nunique()), "n": int(len(g)), **eff})
    return pd.DataFrame(rows)


def placebo(con: duckdb.DuckDBPyConnection, waves: list[Wave], cfg: config.CausalConfig) -> dict:
    """Pretend each wave started halfway through its baseline. Nothing happened then,
    so a working design should find ~0. Matching is redone using only the first half."""
    per_wave, frames = [], []
    for w in waves:
        fake = Wave(w.wave, w.campaign_id, w.pre_start, w.pre_start + max(w.pre_weeks // 2, 2), w.start - 1)
        if fake.start - fake.pre_start < 2 or fake.end < fake.start:
            continue
        res, m = analyze_wave(con, fake, cfg)
        per_wave.append(
            {
                "wave": w.wave,
                "fake_pre": res["pre_window"],
                "fake_post": res["post_window"],
                **res["did_matched"],
            }
        )
        frames.append(m)
    pooled = stacked_effect(pd.concat(frames, ignore_index=True)) if frames else None
    return {"per_wave": per_wave, "pooled": pooled}


def cuped(matched: pd.DataFrame) -> dict:
    """Compare three ways to analyse the same matched sample.

    post-only: post spend difference (ignores the baseline, noisy)
    DiD:       change in spend (subtracts the baseline with weight 1)
    CUPED:     post - theta * (baseline - mean), theta chosen to minimise variance.
    DiD is CUPED with theta fixed at 1, which is why both beat post-only.
    """
    m = matched.copy()
    m["pre_c"] = m.pre_spend - m.groupby("wave").pre_spend.transform("mean")
    m["post_c"] = m.post_spend - m.groupby("wave").post_spend.transform("mean")
    theta = float(np.cov(m.post_c, m.pre_c)[0, 1] / np.var(m.pre_c, ddof=1))
    m["post_cuped"] = m.post_spend - theta * m.pre_c
    post = stacked_effect(m, y="post_spend")
    did = stacked_effect(m, y="dy")
    cup = stacked_effect(m, y="post_cuped")
    rho = float(np.corrcoef(m.post_c, m.pre_c)[0, 1])
    return {
        "theta": theta,
        "rho_pre_post": rho,
        "post_only": post,
        "did": did,
        "cuped": cup,
        "variance_reduction_vs_post": 1 - (cup["se"] / post["se"]) ** 2,
        "sample_size_multiplier": (post["se"] / cup["se"]) ** 2,
    }


def power_analysis(
    matched: pd.DataFrame, effect_per_week: float, alpha: float = 0.05, power: float = 0.8
) -> dict:
    """Size a randomised holdout for the next wave.

    MDE = (z_{1-a/2} + z_{power}) * sigma * sqrt(1/n_t + 1/n_c), with sigma^2 scaled by
    (1 - rho^2) when CUPED uses baseline spend as the covariate.
    """
    t = matched[matched.treated == 1]
    sigma = float(t.post_spend.std(ddof=1))
    rho = float(np.corrcoef(matched.post_spend, matched.pre_spend)[0, 1])
    baseline = float(t.pre_spend.mean())
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    wave_size = int(matched[matched.treated == 1].groupby("wave").size().mean() * 2)
    shares = np.round(np.arange(0.05, 0.51, 0.05), 2)
    curve = []
    for N in (1100, 2200, 4400):
        for h in shares:
            n_c, n_t = h * N, (1 - h) * N
            se_unit = np.sqrt(1 / n_t + 1 / n_c)
            curve.append(
                {
                    "households": N,
                    "holdout_share": float(h),
                    "mde_plain": z * sigma * se_unit,
                    "mde_cuped": z * sigma * np.sqrt(1 - rho**2) * se_unit,
                }
            )
    eff = abs(effect_per_week) if effect_per_week else float("nan")
    n_plain = 2 * (z * sigma / eff) ** 2
    n_cuped = n_plain * (1 - rho**2)
    return {
        "sigma_weekly_spend": sigma,
        "rho": rho,
        "baseline_weekly_spend": baseline,
        "alpha": alpha,
        "power": power,
        "effect_used": effect_per_week,
        "n_per_arm_plain": float(n_plain),
        "n_per_arm_cuped": float(n_cuped),
        "typical_wave_size": wave_size,
        "curve": curve,
    }


def segment_effects(matched: pd.DataFrame) -> list[dict]:
    """Does the effect differ by who the household was before the wave?

    Segments are defined on behaviour from BEFORE the baseline window (prior spend,
    prior coupon use, earlier booklets). Defining them on the baseline window itself
    selects on noise: a household that happened to have a quiet baseline lands in the
    "light" segment and then bounces back (regression to the mean), which looks like a
    big positive effect that isn't real. Wave 1 has no prior window, so it sits out.

    The segment comes from the recipient in each pair and is copied to its match, so
    the comparison inside every segment is still recipient vs look-alike.
    """
    m = matched[matched.prior_weeks >= 4].copy()
    t = m[m.treated == 1].copy()
    t["spend_tier"] = t.groupby("wave").prior_spend.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["light", "medium", "heavy"])
    )
    t["booklet_seg"] = np.where(t.prior_waves > 0, "had a booklet before", "first booklet")
    t["coupon_seg"] = np.where(t.prior_coupon_use > 0, "used coupons before", "never used coupons")
    seg = t.set_index("pair_id")[["spend_tier", "booklet_seg", "coupon_seg"]]
    m = m.join(seg, on="pair_id")
    out = []
    for dim in ["spend_tier", "booklet_seg", "coupon_seg"]:
        for level, g in m.groupby(dim, observed=True):
            if g.treated.sum() < 30:
                continue
            eff = stacked_effect(g)
            base = float(g[g.treated == 1].pre_spend.mean())
            out.append(
                {
                    "dimension": dim,
                    "segment": str(level),
                    "n_pairs": int(g.treated.sum()),
                    "baseline": base,
                    **eff,
                }
            )
    return out


def segment_effects_naive(matched: pd.DataFrame) -> list[dict]:
    """The WRONG way, kept on purpose for the write-up: tiers cut on baseline spend."""
    m = matched.copy()
    t = m[m.treated == 1].copy()
    t["tier"] = t.groupby("wave").pre_spend.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["light", "medium", "heavy"])
    )
    m = m.join(t.set_index("pair_id")[["tier"]], on="pair_id")
    return [
        {"segment": str(k), "n_pairs": int(g.treated.sum()), **stacked_effect(g)}
        for k, g in m.groupby("tier", observed=True)
    ]


# --------------------------------------------------------------------------- orchestration


def run(cfg: config.CausalConfig = config.SETTINGS.causal) -> dict:
    con = connect()
    try:
        waves = list_waves(con, cfg)
        log.info("waves: %s", [w.label for w in waves])
        wave_results, matched_frames = [], []
        for w in waves:
            res, m = analyze_wave(con, w, cfg)
            wave_results.append(res)
            matched_frames.append(m)
            log.info(
                "%s: matched %d/%d recipients, DiD %.2f [%.2f, %.2f] $/wk, max|SMD| after %.3f",
                w.label,
                res["n_pairs"],
                res["n_treated"],
                res["did_matched"]["est"],
                res["did_matched"]["lo"],
                res["did_matched"]["hi"],
                res["max_abs_smd_after"],
            )
        matched = pd.concat(matched_frames, ignore_index=True)
        all_units = pd.concat([unit_table(con, w, cfg) for w in waves], ignore_index=True)

        pooled_did = stacked_effect(matched)
        baseline = float(matched[matched.treated == 1].pre_spend.mean())
        pooled = {
            "naive_post": stacked_effect(all_units, y="post_spend"),
            "did_unmatched": stacked_effect(all_units, y="dy"),
            "did_matched": pooled_did,
            "did_matched_trips": stacked_effect(matched, y="dy_trips"),
            "baseline_weekly_spend": baseline,
            "rel_lift": pooled_did["est"] / baseline,
            "rel_lift_lo": pooled_did["lo"] / baseline,
            "rel_lift_hi": pooled_did["hi"] / baseline,
            "n_pairs": int(matched.treated.sum()),
            "n_treated": int(all_units.treated.sum()),
            "post_weeks_avg": float(np.mean([w.post_weeks for w in waves])),
        }
        # DR: regression adjustment on the matched sample, pooled
        Z = design_matrix(matched, list(cfg.covariates))
        dums = pd.get_dummies(matched["wave"].astype(str), prefix="wave", drop_first=True, dtype=float)
        X = pd.concat([matched[["treated"]].astype(float), Z, dums], axis=1)
        pooled["did_matched_dr"] = ols_effect(matched.dy, X, groups=matched.household_id)

        es = event_study(con, waves, matched)
        plc = placebo(con, waves, cfg)
        robustness = robustness_checks(con, waves, cfg, pooled, plc)
        segs = segment_effects(matched)
        segs_naive = segment_effects_naive(matched)
        cup = cuped(matched)
        pw = power_analysis(matched, pooled_did["est"])
        roi = campaign_roi(pooled)
    finally:
        con.close()

    results = {
        "design": {
            "waves": [asdict(w) | {"label": w.label} for w in waves],
            "covariates": list(cfg.covariates),
            "caliper_sd": cfg.caliper_sd,
            "outcome": "average weekly spend excluding fuel, $",
        },
        "waves": wave_results,
        "pooled": pooled,
        "event_study": es.to_dict("records"),
        "placebo": plc,
        "robustness": robustness,
        "segments": segs,
        "segments_regression_to_mean_trap": segs_naive,
        "cuped": cup,
        "power": pw,
        "roi": roi,
    }
    results["headline"] = headline(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "causal.json").write_text(json.dumps(results, indent=2, default=float))
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(config.ARTIFACTS_DIR / "causal_matched.parquet", index=False)
    make_figures(results)
    log.info(results["headline"])
    return results


def robustness_checks(con, waves, cfg, pooled, plc) -> list[dict]:
    specs: list[dict] = [{"spec": "main: matched DiD", **pooled["did_matched"], "n_pairs": pooled["n_pairs"]}]
    specs.append(
        {
            "spec": "+ regression adjustment (doubly robust)",
            **pooled["did_matched_dr"],
            "n_pairs": pooled["n_pairs"],
        }
    )

    def rerun(name: str, cfg_override: dict | None = None, **kw) -> None:
        c = replace(cfg, **cfg_override) if cfg_override else cfg
        frames = [analyze_wave(con, w, c, **kw)[1] for w in waves]
        m = pd.concat(frames, ignore_index=True)
        specs.append({"spec": name, **stacked_effect(m), "n_pairs": int(m.treated.sum())})

    rerun("looser caliper (0.2 SD)", caliper_sd=0.2)
    rerun("matching with replacement", cfg_override={"with_replacement": True})
    rerun("no other campaigns during the wave", subset=lambda d: d[d.co_exposure_post == 0])
    rerun("propensity only (no exact matching)", exact_on=[])
    rerun(
        "spend + trips only (fewer covariates)",
        covariates=["pre_spend", "pre_trips", "pre_active_share", "prior_spend"],
    )
    drd = dr_did(con, waves, cfg)
    specs.append({"spec": "DR-DiD weighting, no matching (Sant'Anna-Zhao)", **drd, "n_pairs": None})
    if plc.get("pooled"):
        specs.append({"spec": "placebo: fake start mid-baseline", **plc["pooled"], "n_pairs": None})
    specs.append({"spec": "unmatched DiD (no matching)", **pooled["did_unmatched"], "n_pairs": None})
    return specs


def campaign_roi(pooled: dict, biz: config.BusinessConfig = config.SETTINGS.business) -> dict:
    """Did a booklet pay for itself? Break-even lift = cost / (weeks x margin).

    sales_value is what the retailer actually receives (after its own discounts), so the
    incremental margin is lift x weeks x gross margin. The probability of profit uses
    the normal approximation to the matched DiD estimate. All money here is simulated
    from the stated margin and cost assumptions; the data has neither.
    """
    d = pooled["did_matched"]
    weeks = pooled["post_weeks_avg"]
    breakeven = biz.cost_per_contact / (weeks * biz.gross_margin)
    profit = d["est"] * weeks * biz.gross_margin - biz.cost_per_contact
    p_profit = float(1 - stats.norm.cdf((breakeven - d["est"]) / d["se"]))
    grid = []
    for m in biz.margin_grid:
        for c in biz.cost_grid:
            be = c / (weeks * m)
            grid.append(
                {
                    "margin": m,
                    "cost": c,
                    "breakeven_lift": be,
                    "p_profit": float(1 - stats.norm.cdf((be - d["est"]) / d["se"])),
                }
            )
    return {
        "assumptions": {
            "gross_margin": biz.gross_margin,
            "cost_per_booklet": biz.cost_per_contact,
            "weeks": weeks,
        },
        "breakeven_lift_per_week": breakeven,
        "breakeven_rel": breakeven / pooled["baseline_weekly_spend"],
        "simulated_profit_per_booklet": profit,
        "simulated_profit_lo": d["lo"] * weeks * biz.gross_margin - biz.cost_per_contact,
        "simulated_profit_hi": d["hi"] * weeks * biz.gross_margin - biz.cost_per_contact,
        "prob_profitable": p_profit,
        "sensitivity": grid,
    }


def headline(r: dict) -> str:
    p = r["pooled"]
    d = p["did_matched"]
    naive = p["naive_post"]["est"]
    return (
        f"Recipients spent ${naive:.0f}/week more than non-recipients during the waves, but after matching on "
        f"pre-wave behaviour the booklets' effect was {d['est']:+.2f} $/week "
        f"({p['rel_lift']:+.1%}, 95% CI {p['rel_lift_lo']:+.1%} to {p['rel_lift_hi']:+.1%}) "
        f"across {p['n_pairs']:,} matched pairs in {len(r['waves'])} waves."
    )


# --------------------------------------------------------------------------- figures

COV_LABELS = {
    "pre_spend": "weekly spend",
    "pre_trips": "weekly trips",
    "pre_active_share": "share of weeks shopping",
    "pre_departments": "departments per week",
    "pre_private_share": "store-brand share",
    "pre_promo_share": "promo-item share",
    "pre_coupon_use": "coupon use",
    "pre_trend": "spend trend",
    "prior_spend": "spend before baseline",
    "prior_waves": "earlier booklets",
    "co_exposure_pre": "other campaigns (baseline)",
    "co_exposure_post": "other campaigns (during)",
}


def make_figures(r: dict) -> None:
    _fig_balance(r)
    _fig_estimates(r)
    _fig_event_study(r)
    _fig_robustness(r)
    _fig_cuped_power(r)
    _fig_segments(r)


def _fig_balance(r: dict) -> None:
    waves = r["waves"]
    fig, axes = plt.subplots(1, len(waves), figsize=(3.1 * len(waves) + 1.5, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    covs = [b["covariate"] for b in waves[0]["balance"]]
    y = np.arange(len(covs))
    for ax, w in zip(axes, waves, strict=True):
        b = pd.DataFrame(w["balance"]).set_index("covariate").loc[covs]
        ax.axvspan(-0.1, 0.1, color=P.MID, zorder=0)
        ax.axvline(0, color=P.AXIS, lw=0.8)
        for yi, (bf, af) in enumerate(zip(b.before, b.after, strict=True)):
            ax.plot([bf, af], [yi, yi], color=P.GRID, lw=1.2, zorder=1)
        ax.scatter(
            b.before,
            y,
            s=34,
            color=P.NEUTRAL,
            zorder=2,
            label="before matching",
            edgecolor=P.SURFACE,
            linewidth=1.5,
        )
        ax.scatter(
            b.after,
            y,
            s=34,
            color=P.BLUE,
            zorder=3,
            label="after matching",
            edgecolor=P.SURFACE,
            linewidth=1.5,
        )
        ax.set_title(f"wave {w['wave']}", fontsize=10)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("standardised mean difference")
    axes[0].set_yticks(y, [COV_LABELS.get(c, c) for c in covs])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Matching removes the gap between recipients and non-recipients",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.985,
        "grey band = |SMD| < 0.1, the usual bar for good balance",
        fontsize=9,
        color=P.INK_2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.save(fig, "causal_1_balance")


def _fig_estimates(r: dict) -> None:
    waves, pooled = r["waves"], r["pooled"]
    labels = [f"wave {w['wave']}" for w in waves] + ["all waves"]
    naive = [w["naive_post"]["est"] for w in waves] + [pooled["naive_post"]["est"]]
    did = [w["did_matched"] for w in waves] + [pooled["did_matched"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(labels))
    a1.bar(x - 0.17, naive, width=0.32, color=P.NEUTRAL, label="naive: recipients minus non-recipients")
    a1.bar(x + 0.17, [d["est"] for d in did], width=0.32, color=P.BLUE, label="matched DiD (the real effect)")
    for xi, v in zip(x, naive, strict=True):
        a1.text(xi - 0.17, v + 0.6, f"${v:.0f}", ha="center", fontsize=8.5, color=P.INK_2)
    for xi, d in zip(x, did, strict=True):
        a1.text(
            xi + 0.17, max(d["est"], 0) + 0.6, P.money(d["est"], 1), ha="center", fontsize=8.5, color=P.INK
        )
    a1.set_xticks(x, labels)
    a1.set_ylabel("$ per household per week")
    a1.set_ylim(min(0, min(d["est"] for d in did)) - 1, max(naive) * 1.25)
    a1.set_title("The naive gap is mostly targeting bias")
    P.subtitle(a1, "difference in weekly spend during the wave")
    a1.legend(loc="upper left", fontsize=8.5)

    y = np.arange(len(labels))[::-1]
    unm = [w["did_unmatched"] for w in waves] + [pooled["did_unmatched"]]
    for yi, d, u in zip(y, did, unm, strict=True):
        a2.plot([u["lo"], u["hi"]], [yi + 0.15, yi + 0.15], color=P.NEUTRAL, lw=2)
        a2.plot(u["est"], yi + 0.15, "o", color=P.NEUTRAL, mec=P.SURFACE, mew=1.5, ms=7)
        a2.plot([d["lo"], d["hi"]], [yi - 0.15, yi - 0.15], color=P.BLUE, lw=2)
        a2.plot(d["est"], yi - 0.15, "o", color=P.BLUE, mec=P.SURFACE, mew=1.5, ms=8 if yi else 10)
    a2.axvline(0, color=P.INK_2, lw=0.9)
    a2.set_yticks(y, labels)
    a2.grid(axis="x")
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("effect on weekly spend, $ (95% CI)")
    a2.set_title("Effect estimates with uncertainty")
    P.subtitle(a2, "grey = DiD without matching, blue = matched DiD")
    fig.tight_layout()
    P.save(fig, "causal_2_estimates")


def _fig_event_study(r: dict) -> None:
    es = pd.DataFrame(r["event_study"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    post_end = es.rel_week.max()
    ax.axvspan(-0.5, post_end + 0.5, color=P.BLUE, alpha=0.05, lw=0)
    ax.axhline(0, color=P.INK_2, lw=0.9)
    ax.fill_between(es.rel_week, es.lo, es.hi, color=P.BLUE, alpha=0.12, lw=0, step=None)
    ax.plot(es.rel_week, es.est, color=P.BLUE)
    ax.plot(es.rel_week, es.est, "o", color=P.BLUE, mec=P.SURFACE, mew=1.5, ms=6)
    ax.text(0, ax.get_ylim()[1] * 0.92, "  booklet in hand", fontsize=9, color=P.INK_2, va="top")
    ax.text(-0.6, ax.get_ylim()[1] * 0.92, "baseline  ", fontsize=9, color=P.INK_2, va="top", ha="right")
    ax.set_xlabel("weeks relative to the start of the wave")
    ax.set_ylabel("recipient minus match, $/week")
    ax.set_title("Event study: flat before the wave, small and noisy during it")
    P.subtitle(ax, "weekly spend gap vs each household's own baseline average, all waves pooled, 95% CI")
    ax.grid(axis="x")
    P.save(fig, "causal_3_event_study")


def _fig_robustness(r: dict) -> None:
    rob = pd.DataFrame(r["robustness"])
    fig, ax = plt.subplots(figsize=(8.6, 0.48 * len(rob) + 1.6))
    y = np.arange(len(rob))[::-1]
    for yi, row in zip(y, rob.itertuples(), strict=True):
        is_main = row.spec.startswith("main")
        is_check = row.spec.startswith("placebo") or row.spec.startswith("unmatched")
        color = P.NEUTRAL if is_check else P.BLUE
        ax.plot([row.lo, row.hi], [yi, yi], color=color, lw=2)
        ax.plot(row.est, yi, "o", color=color, mec=P.SURFACE, mew=1.5, ms=9 if is_main else 7)
        ax.text(row.hi + 0.15, yi, f"{row.est:+.2f}", va="center", fontsize=8.5, color=P.INK_2)
    ax.axvline(0, color=P.INK_2, lw=0.9)
    ax.set_yticks(y, rob.spec)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect on weekly spend, $ (95% CI), all waves pooled")
    ax.set_title("Robustness checks")
    P.subtitle(
        ax, "the estimate should not swing wildly between reasonable choices; the placebo should sit at 0"
    )
    P.save(fig, "causal_4_robustness")


def _fig_cuped_power(r: dict) -> None:
    c, p = r["cuped"], r["power"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    names = ["post spend only", "DiD (theta = 1)", f"CUPED (theta = {c['theta']:.2f})"]
    hw = [Z95 * c[k]["se"] for k in ("post_only", "did", "cuped")]
    a1.barh(names[::-1], hw[::-1], color=[P.BLUE, P.BLUE, P.NEUTRAL], height=0.5)
    for i, v in enumerate(hw[::-1]):
        a1.text(v + 0.05, i, f"±${v:.2f}", va="center", fontsize=9, color=P.INK)
    a1.set_xlim(0, max(hw) * 1.3)
    a1.grid(axis="x")
    a1.grid(axis="y", visible=False)
    a1.set_title("Using the baseline shrinks the noise")
    P.subtitle(
        a1,
        f"95% CI half-width, same matched sample (CUPED cuts variance {c['variance_reduction_vs_post']:.0%})",
    )

    curve = pd.DataFrame(p["curve"])
    base = p["baseline_weekly_spend"]
    for i, N in enumerate(sorted(curve.households.unique())):
        d = curve[curve.households == N]
        a2.plot(
            d.holdout_share, 100 * d.mde_cuped / base, color=P.SERIES[i], label=f"{N:,} households, CUPED"
        )
        a2.plot(d.holdout_share, 100 * d.mde_plain / base, color=P.SERIES[i], lw=1.2, alpha=0.45)
    a2.axhline(100 * r["pooled"]["rel_lift"], color=P.INK_2, lw=0.9)
    a2.text(
        0.5,
        100 * r["pooled"]["rel_lift"],
        " effect we estimated",
        fontsize=8.5,
        color=P.INK_2,
        va="bottom",
        ha="right",
    )
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_xlabel("share of households held out (no booklet)")
    a2.set_ylabel("minimum detectable lift, % of spend")
    a2.set_ylim(0, None)
    a2.set_title("How big a holdout test would need to be")
    P.subtitle(a2, "80% power, 5% significance; faint lines = without CUPED")
    a2.legend(fontsize=8.5)
    fig.tight_layout()
    P.save(fig, "causal_5_cuped_power")


def _fig_segments(r: dict) -> None:
    seg = pd.DataFrame(r["segments"])
    if seg.empty:
        return
    titles = {
        "spend_tier": "spend tier (before baseline)",
        "booklet_seg": "booklet history",
        "coupon_seg": "coupon habit",
    }
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(seg) + 1.6))
    y = np.arange(len(seg))[::-1]
    labels = [f"{titles[d]}: {s}" for d, s in zip(seg.dimension, seg.segment, strict=True)]
    for yi, row in zip(y, seg.itertuples(), strict=True):
        ax.plot([row.lo, row.hi], [yi, yi], color=P.BLUE, lw=2)
        ax.plot(row.est, yi, "o", color=P.BLUE, mec=P.SURFACE, mew=1.5, ms=7)
        ax.text(
            row.hi + 0.2, yi, f"{row.est:+.2f}  (n={row.n_pairs:,})", va="center", fontsize=8.5, color=P.INK_2
        )
    ax.axvline(0, color=P.INK_2, lw=0.9)
    ax.axvline(r["pooled"]["did_matched"]["est"], color=P.NEUTRAL, lw=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect on weekly spend, $ (95% CI)")
    ax.set_title("Who responds? Effect by segment")
    P.subtitle(ax, "matched DiD inside each segment, waves 2-4 pooled; grey line = overall effect")
    P.save(fig, "causal_6_segments")
