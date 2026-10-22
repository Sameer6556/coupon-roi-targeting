"""Who would a booklet actually change? Uplift (conditional treatment effect) models.

A churn score answers "who is at risk". The targeting question is different: "whose
spend goes up BECAUSE they got the booklet". High-risk households may not respond at
all, and loyal heavy shoppers would buy anyway. Uplift models estimate the effect for
each household:  tau(x) = E[spend change | booklet, x] - E[spend change | no booklet, x].

Three learners, all on LightGBM:
  T-learner   separate outcome models for recipients and non-recipients, take the gap
  X-learner   imputes each side's counterfactual, then blends by propensity
  DR-learner  doubly robust pseudo-outcomes with cross-fitting (robust to targeting bias)

Two honesty checks before anything gets used for targeting:
  1. Semi-synthetic benchmark: real households and the real targeting rule, but outcomes
     simulated with KNOWN effects. If a learner can't find planted heterogeneity, its
     real-data output means nothing.
  2. Out-of-time check on real data: train on earlier waves, score the next wave, and
     compare predictions with what actually happened in the matched pairs. The slope of
     realised-on-predicted effects says how much to trust (shrink) the predictions.
"""

from __future__ import annotations

import json
import logging
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config, tracking
from . import plotting as P
from .causal import list_waves, unit_table
from .churn import LogitModel
from .features import FEATURES, features_at
from .warehouse import connect

log = logging.getLogger(__name__)

REG_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": 300,
    "num_leaves": 7,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
}
FINAL_PARAMS = {**REG_PARAMS, "n_estimators": 200, "min_child_samples": 200}
PS_CLIP = (0.05, 0.95)


def _reg(params: dict = REG_PARAMS, seed: int = config.SEED) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**params, random_state=seed)


def _fit(model, X: pd.DataFrame, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y)


class Propensity:
    """P(booklet | features). Logistic regression: smooth, and hard to overfit."""

    def fit(self, X: pd.DataFrame, t: np.ndarray) -> Propensity:
        self.m = LogitModel().fit(X, t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.m.predict(X), *PS_CLIP)


class TLearner:
    name = "T-learner"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> TLearner:
        self.m1 = _fit(_reg(seed=seed), X[t == 1], y[t == 1])
        self.m0 = _fit(_reg(seed=seed), X[t == 0], y[t == 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    name = "X-learner"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> XLearner:
        m1 = _fit(_reg(seed=seed), X[t == 1], y[t == 1])
        m0 = _fit(_reg(seed=seed), X[t == 0], y[t == 0])
        d1 = y[t == 1] - m0.predict(X[t == 1])  # recipients: actual minus imputed no-booklet outcome
        d0 = m1.predict(X[t == 0]) - y[t == 0]  # non-recipients: imputed booklet outcome minus actual
        self.tau1 = _fit(_reg(FINAL_PARAMS, seed), X[t == 1], d1)
        self.tau0 = _fit(_reg(FINAL_PARAMS, seed), X[t == 0], d0)
        self.g = Propensity().fit(X, t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        g = self.g.predict(X)
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


class DRLearner:
    """Doubly robust learner (Kennedy 2020) with K-fold cross-fitting.

    pseudo-outcome  phi = m1(x) - m0(x) + t (y - m1(x)) / e(x) - (1 - t)(y - m0(x)) / (1 - e(x))
    Each household's phi uses nuisance models fitted on the OTHER folds, so a household
    never helps predict its own pseudo-outcome. Then regress phi on x.
    """

    name = "DR-learner"

    def __init__(self, folds: int = 3):
        self.folds = folds

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> DRLearner:
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, self.folds, size=len(X))
        phi = np.zeros(len(X))
        for k in range(self.folds):
            tr, te = fold != k, fold == k
            Xtr, ttr, ytr = X[tr], t[tr], y[tr]
            m1 = _fit(_reg(seed=seed), Xtr[ttr == 1], ytr[ttr == 1])
            m0 = _fit(_reg(seed=seed), Xtr[ttr == 0], ytr[ttr == 0])
            e = Propensity().fit(Xtr, ttr).predict(X[te])
            mu1, mu0 = m1.predict(X[te]), m0.predict(X[te])
            phi[te] = mu1 - mu0 + t[te] * (y[te] - mu1) / e - (1 - t[te]) * (y[te] - mu0) / (1 - e)
        self.final = _fit(_reg(FINAL_PARAMS, seed), X, phi)
        self.phi_ = phi
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final.predict(X)


LEARNERS = {"T-learner": TLearner, "X-learner": XLearner, "DR-learner": DRLearner}


# --------------------------------------------------------------------------- data


def winsor_bounds(y: np.ndarray, q: float = 0.01) -> tuple[float, float]:
    return float(np.quantile(y, q)), float(np.quantile(y, 1 - q))


def wave_frame(con, cfg: config.Settings = config.SETTINGS) -> pd.DataFrame:
    """One row per household per wave: features as of the week before the wave, whether it
    got the booklet, and its change in weekly spend (campaign weeks minus baseline)."""
    frames = []
    for w in list_waves(con, cfg.causal):
        cutoff = w.start - 1
        if cutoff < cfg.churn.first_cutoff:  # not enough history for 16-week features
            continue
        f = features_at(con, cutoff)
        u = unit_table(con, w, cfg.causal)[["household_id", "treated", "dy", "pre_spend", "post_spend"]]
        d = f.merge(u, on="household_id", how="inner")
        d["wave"] = w.wave
        d["campaign_id"] = w.campaign_id
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- semi-synthetic benchmark


def planted_effect(X: pd.DataFrame) -> np.ndarray:
    """The 'true' per-household effect used in the benchmark, in $ per week.

    Coupon users and households whose spend has been sliding respond more; this is a
    made-up but plausible pattern, scaled so its spread is about $3/week.
    """
    coupon_user = (X["redemptions_16w"] > 0).astype(float).to_numpy()
    sliding = np.tanh(-8 * X["spend_slope_rel_16w"].fillna(0).to_numpy())
    cooling = (X["momentum_4v16"].fillna(1).to_numpy() < 0.8).astype(float)
    raw = 2.5 * coupon_user + 2.0 * sliding + 1.5 * cooling
    return 1.0 + 3.0 * (raw - raw.mean()) / raw.std()


def semi_synthetic(frame: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2), top_share: float = 0.3) -> pd.DataFrame:
    X = frame[FEATURES].reset_index(drop=True)
    t_real = frame["treated"].to_numpy()
    lo, hi = winsor_bounds(frame["dy"].to_numpy())
    y_real = frame["dy"].clip(lo, hi).to_numpy()

    base = _fit(_reg(), X[t_real == 0], y_real[t_real == 0])
    mu0 = base.predict(X)
    resid = y_real[t_real == 0] - base.predict(X[t_real == 0])
    tau = planted_effect(X)
    e_real = Propensity().fit(X, t_real).predict(X)

    rows = []
    for scenario, e in [("randomised (like an A/B test)", np.full(len(X), 0.5)), ("targeted (like the real booklets)", e_real)]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            t = rng.binomial(1, e)
            y = mu0 + t * tau + rng.choice(resid, size=len(X), replace=True)
            idx = rng.permutation(len(X))
            tr, te = idx[: int(0.7 * len(X))], idx[int(0.7 * len(X)) :]
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            oracle_top = np.sort(tau[te])[::-1][: int(top_share * len(te))].mean()
            naive = y[t == 1].mean() - y[t == 0].mean()
            for name, cls in LEARNERS.items():
                model = cls().fit(Xtr, t[tr], y[tr], seed=seed)
                pred = model.predict(Xte)
                top = np.argsort(-pred)[: int(top_share * len(te))]
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "learner": name,
                        "pehe": float(np.sqrt(np.mean((pred - tau[te]) ** 2))),
                        "spearman": float(stats.spearmanr(pred, tau[te]).statistic),
                        "ate_error": float(pred.mean() - tau[te].mean()),
                        "policy_share_of_oracle": float(tau[te][top].mean() / oracle_top),
                        "random_share_of_oracle": float(tau[te].mean() / oracle_top),
                        "naive_ate_error": float(naive - tau.mean()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- out-of-time check


def pair_table(matched: pd.DataFrame, wave: int, lo: float, hi: float) -> pd.DataFrame:
    """One row per matched pair: realised effect = recipient's spend change minus its match's."""
    m = matched[matched.wave == wave]
    t = m[m.treated == 1].set_index("pair_id")
    c = m[m.treated == 0].set_index("pair_id")
    pairs = pd.DataFrame(
        {
            "household_id": t.household_id,
            "control_id": c.household_id.reindex(t.index),
            "delta": t.dy.clip(lo, hi) - c.dy.reindex(t.index).clip(lo, hi),
        }
    ).dropna()
    pairs["wave"] = wave
    return pairs.reset_index()


def out_of_time(frame: pd.DataFrame, matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward chaining: train on earlier waves, score the next one, check against its pairs."""
    waves = sorted(frame.wave.unique())
    all_pairs, rows = [], []
    for i in range(1, len(waves)):
        train_w, test_w = waves[:i], waves[i]
        tr = frame[frame.wave.isin(train_w)]
        lo, hi = winsor_bounds(tr["dy"].to_numpy())  # outlier caps from training waves only
        te = frame[frame.wave == test_w].set_index("household_id")
        pairs = pair_table(matched, test_w, lo, hi)
        pairs = pairs[pairs.household_id.isin(te.index)]
        Xp = te.loc[pairs.household_id, FEATURES]
        for name, cls in LEARNERS.items():
            model = cls().fit(tr[FEATURES], tr.treated.to_numpy(), tr.dy.clip(lo, hi).to_numpy())
            pairs[name] = model.predict(Xp)
        pairs["heaviest shoppers"] = te.loc[pairs.household_id, "spend_16w"].to_numpy()
        pairs["random"] = np.random.default_rng(config.SEED + i).random(len(pairs))
        pairs["train_waves"] = ",".join(map(str, train_w))
        all_pairs.append(pairs)
    pairs = pd.concat(all_pairs, ignore_index=True)
    for name in LEARNERS:
        rows.append({"learner": name, **calibration_slope(pairs, name), **top_share_gain(pairs, name)})
    for name in ("heaviest shoppers", "random"):
        rows.append({"learner": name, **top_share_gain(pairs, name)})
    return pairs, pd.DataFrame(rows)


def calibration_slope(pairs: pd.DataFrame, col: str) -> dict:
    """Regress realised pair effects on predicted effects. Slope 1 = predictions are right
    on average; 0 = they carry no information. Used to shrink predictions for targeting."""
    X = sm.add_constant(pairs[[col]].astype(float))
    fit = sm.OLS(pairs["delta"].astype(float), X).fit(cov_type="HC1")
    return {"slope": float(fit.params[col]), "slope_se": float(fit.bse[col]), "slope_p": float(fit.pvalues[col])}


def top_share_gain(pairs: pd.DataFrame, col: str, share: float = 0.3, reps: int = 500) -> dict:
    """Mean realised effect in the top 30% of pairs ranked by `col`, vs all pairs."""
    rng = np.random.default_rng(config.SEED)
    k = int(share * len(pairs))
    order = np.argsort(-pairs[col].to_numpy(), kind="stable")
    delta = pairs["delta"].to_numpy()
    top = delta[order[:k]].mean()
    boots = []
    for _ in range(reps):
        b = rng.integers(0, len(pairs), len(pairs))
        d, s = delta[b], pairs[col].to_numpy()[b]
        o = np.argsort(-s, kind="stable")
        boots.append(d[o[:k]].mean() - d.mean())
    return {
        "top30_effect": float(top),
        "all_effect": float(delta.mean()),
        "top30_gain": float(top - delta.mean()),
        "top30_gain_lo": float(np.quantile(boots, 0.025)),
        "top30_gain_hi": float(np.quantile(boots, 0.975)),
        "n_pairs": int(len(pairs)),
    }


def uplift_curve(pairs: pd.DataFrame, col: str) -> pd.DataFrame:
    order = np.argsort(-pairs[col].to_numpy(), kind="stable")
    d = pairs["delta"].to_numpy()[order]
    n = len(d)
    return pd.DataFrame({"share": np.arange(1, n + 1) / n, "cum_incremental": np.cumsum(d) / n})


# --------------------------------------------------------------------------- final scoring


def score_latest(con, frame: pd.DataFrame, learner: str, ate: float, slope: float) -> pd.DataFrame:
    """Fit the chosen learner on every wave and score all active households for the NEXT wave.

    tau_shrunk = ATE + clip(slope, 0, 1) x (tau_hat - mean tau_hat). With slope ~0 every
    household gets the average effect, i.e. the model gets no say in who is targeted.
    """
    lo, hi = winsor_bounds(frame["dy"].to_numpy())
    model = LEARNERS[learner]().fit(frame[FEATURES], frame.treated.to_numpy(), frame.dy.clip(lo, hi).to_numpy())
    last_week = int(con.execute("SELECT max(week) FROM dim_week").fetchone()[0])
    f = features_at(con, last_week)
    f["uplift_raw"] = model.predict(f[FEATURES])
    lam = float(np.clip(slope, 0.0, 1.0))
    center = float(f["uplift_raw"].mean())
    f["uplift"] = shrink(f["uplift_raw"].to_numpy(), ate, lam, center)
    f.attrs["shrinkage"] = lam
    save_model(model, {"learner": learner, "ate": ate, "shrinkage": lam, "center": center,
                       "scored_week": last_week, "winsor": [lo, hi], "features": FEATURES})
    return f


def shrink(raw: np.ndarray, ate: float, lam: float, center: float) -> np.ndarray:
    """Pull raw predictions toward the average effect by however much they held up out of time."""
    return ate + lam * (np.asarray(raw) - center)


def save_model(model, meta: dict) -> None:
    import joblib

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "uplift_model.joblib")
    (config.MODEL_DIR / "uplift_meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- figures


def _fig_semisynthetic(bench: pd.DataFrame) -> None:
    agg = bench.groupby(["scenario", "learner"]).agg(
        share=("policy_share_of_oracle", "mean"), rnd=("random_share_of_oracle", "mean"), rho=("spearman", "mean")
    ).reset_index()
    scen = list(dict.fromkeys(bench.scenario))
    learners = list(LEARNERS)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(scen))
    w = 0.22
    for i, name in enumerate(learners):
        v = [agg[(agg.scenario == s) & (agg.learner == name)].share.iloc[0] for s in scen]
        ax.bar(x + (i - 1) * (w + 0.02), v, width=w, color=P.SERIES[i], label=name)
        for xi, vi in zip(x, v, strict=True):
            ax.text(xi + (i - 1) * (w + 0.02), vi + 0.015, f"{vi:.0%}", ha="center", fontsize=8.5, color=P.INK)
    rnd = [agg[agg.scenario == s].rnd.iloc[0] for s in scen]
    for xi, r in zip(x, rnd, strict=True):
        ax.plot([xi - 0.4, xi + 0.4], [r, r], color=P.INK_2, lw=1)
        ax.text(xi + 0.41, r, "random", fontsize=8, color=P.INK_2, va="center")
    ax.set_xticks(x, scen)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("share of the best possible uplift")
    ax.set_title("Benchmark with planted effects: can the learners find them?")
    P.subtitle(ax, "targeting the top 30% by each model, held-out households, average of 3 simulations")
    ax.legend(loc="upper right", ncols=3, fontsize=8.5)
    P.save(fig, "uplift_1_semisynthetic")


def _fig_out_of_time(pairs: pd.DataFrame, oot: pd.DataFrame, chosen: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    for col, color, lw in [(chosen, P.BLUE, 2), ("heaviest shoppers", P.ORANGE, 2), ("random", P.NEUTRAL, 1.5)]:
        c = uplift_curve(pairs, col)
        a1.plot(c.share, c.cum_incremental, color=color, lw=lw, label=col if col != chosen else f"{col} (uplift model)")
    a1.plot([0, 1], [0, pairs.delta.mean()], color=P.INK_2, lw=0.8)
    a1.axhline(0, color=P.AXIS, lw=0.8)
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_xlabel("share of recipients targeted, ranked by each score")
    a1.set_ylabel("realised extra spend per pair, $/week")
    a1.set_title("Out-of-time check on real matched pairs")
    P.subtitle(a1, "uplift curves on waves the models never saw (trained on earlier waves)")
    a1.legend(fontsize=8.5, loc="upper left")
    a1.grid(axis="x")

    rows = oot.set_index("learner").reindex(list(LEARNERS) + ["heaviest shoppers"])
    y = np.arange(len(rows))[::-1]
    for yi, (name, r) in zip(y, rows.iterrows(), strict=True):
        color = P.BLUE if name in LEARNERS else P.ORANGE
        a2.plot([r.top30_gain_lo, r.top30_gain_hi], [yi, yi], color=color, lw=2)
        a2.plot(r.top30_gain, yi, "o", color=color, mec=P.SURFACE, mew=1.5, ms=8)
        label = f"{r.top30_gain:+.2f}" + (f"   slope {r.slope:+.2f}" if name in LEARNERS else "")
        a2.text(r.top30_gain_hi + 0.3, yi, label, fontsize=8.5, color=P.INK_2, va="center")
    a2.axvline(0, color=P.INK_2, lw=0.9)
    a2.set_yticks(y, rows.index)
    a2.grid(axis="x")
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("top-30% effect minus average effect, $/week (95% CI)")
    a2.set_title("Does ranking by the model find bigger effects?")
    P.subtitle(a2, "0 = no better than mailing at random")
    fig.tight_layout()
    P.save(fig, "uplift_2_out_of_time")


# --------------------------------------------------------------------------- run


def choose_learner(bench: pd.DataFrame) -> str:
    """Pick the learner up front by a fixed rule: best share of the oracle uplift in the
    TARGETED benchmark scenario (the one that looks like reality). Choosing by the noisy
    real out-of-time numbers would just be fitting to noise."""
    t = bench[bench.scenario.str.startswith("targeted")]
    return str(t.groupby("learner").policy_share_of_oracle.mean().idxmax())


def run(track: bool = True, chosen: str | None = None) -> dict:
    con = connect()
    try:
        frame = wave_frame(con)
        matched = pd.read_parquet(config.ARTIFACTS_DIR / "causal_matched.parquet")
        causal = json.loads((config.RESULTS_DIR / "causal.json").read_text())
        ate = float(causal["pooled"]["did_matched"]["est"])
        log.info("uplift frame: %d household-waves, %d recipients", len(frame), int(frame.treated.sum()))

        bench = semi_synthetic(frame)
        chosen = chosen or choose_learner(bench)
        log.info("chosen learner: %s", chosen)
        pairs, oot = out_of_time(frame, matched)
        slope = float(oot.set_index("learner").loc[chosen, "slope"])
        scores = score_latest(con, frame, chosen, ate, slope)
    finally:
        con.close()

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(config.ARTIFACTS_DIR / "uplift_oot_pairs.parquet", index=False)
    scores.to_parquet(config.ARTIFACTS_DIR / "uplift_scores_latest.parquet", index=False)
    _fig_semisynthetic(bench)
    _fig_out_of_time(pairs, oot, chosen)

    bench_summary = bench.groupby(["scenario", "learner"])[
        ["pehe", "spearman", "ate_error", "policy_share_of_oracle", "random_share_of_oracle", "naive_ate_error"]
    ].mean().round(4).reset_index()
    chosen_row = oot.set_index("learner").loc[chosen]
    results = {
        "outcome": "change in avg weekly spend ex fuel (campaign weeks minus baseline), winsorised at 1%/99%",
        "n_household_waves": int(len(frame)),
        "waves_used": sorted(int(w) for w in frame.wave.unique()),
        "semi_synthetic": bench_summary.to_dict("records"),
        "out_of_time": oot.round(4).to_dict("records"),
        "chosen_learner": chosen,
        "shrinkage_slope": slope,
        "shrinkage_used": scores.attrs.get("shrinkage"),
        "ate_used": ate,
        "scored_households": int(len(scores)),
        "uplift_spread_raw_sd": float(scores.uplift_raw.std()),
        "uplift_spread_shrunk_sd": float(scores.uplift.std()),
    }
    synth = bench_summary[(bench_summary.learner == chosen) & bench_summary.scenario.str.startswith("targeted")].iloc[0]
    results["headline"] = (
        f"With planted effects and the real (non-random) targeting, the {chosen} captured "
        f"{synth.policy_share_of_oracle:.0%} of the best possible uplift in its top 30% (random: {synth.random_share_of_oracle:.0%}). "
        f"On real held-out waves its predictions had a calibration slope of {slope:.2f} "
        f"(top-30% gain {chosen_row.top30_gain:+.2f} $/week, 95% CI {chosen_row.top30_gain_lo:+.2f} to {chosen_row.top30_gain_hi:+.2f}), "
        "so predictions are shrunk toward the average effect before they drive any targeting."
    )
    with tracking.run("uplift", enabled=track, tags={"stage": "uplift"}) as ml:
        if ml:
            ml.log_params({"chosen_learner": chosen, **{k: v for k, v in REG_PARAMS.items() if k != "verbose"}})
            ml.log_metrics({"oot_slope": slope, "oot_top30_gain": float(chosen_row.top30_gain),
                            "synthetic_policy_share": float(synth.policy_share_of_oracle)})
            for f in sorted(config.FIG_DIR.glob("uplift_*.png")):
                ml.log_artifact(str(f), artifact_path="figures")
    (config.RESULTS_DIR / "uplift.json").write_text(json.dumps(results, indent=2, default=float))
    log.info(results["headline"])
    return results
