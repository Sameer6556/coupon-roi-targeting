"""Lapse-risk model: which regular shoppers are about to go quiet in the next 8 weeks?

Validation mimics deployment. For each test week, the model is trained only on
cutoffs whose labels were fully known by then, calibrated on the most recent of
those, and scored on the test cutoff (rolling-origin backtest):

    train cutoffs  ->  calibration cutoffs  ->  test cutoff        (time flows right)

Baselines: a rule an analyst would write on day one (spend dropped in the last
4 weeks), and a logistic regression. The score that ships is LightGBM with Platt
calibration, because the targeting step multiplies probabilities by dollars.
"""

from __future__ import annotations

import json
import logging
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config, tracking
from . import plotting as P
from .features import FEATURE_DESCRIPTIONS, FEATURES, features_at, is_regular, training_frame
from .models import IsotonicCalibrator, LogitModel, PlattCalibrator
from .warehouse import connect

log = logging.getLogger(__name__)
TARGET = "lapse"
# Small, heavily regularised trees: ~1,400 regulars per cutoff and a noisy target
# reward simple models (a bigger tree setup did no better than logistic regression).
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "n_estimators": 2000,
    "verbose": -1,
    # same data + same params -> same model, whatever the thread count (LightGBM is not
    # deterministic by default; reruns drifted in the 3rd decimal before this)
    "deterministic": True,
    "force_row_wise": True,
}


# --------------------------------------------------------------------------- models


def heuristic_score(df: pd.DataFrame) -> np.ndarray:
    """Day-one analyst rule: rank by how far the last 4 weeks fell below the 16-week average,
    with recency as a tie-breaker."""
    momentum = df["momentum_4v16"].fillna(1.0).clip(0, 3)
    return (-momentum + 0.05 * df["recency_weeks"]).to_numpy()


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame, seed: int = config.SEED) -> lgb.LGBMClassifier:
    """Early stopping on the calibration cutoffs, which are earlier in time than the test."""
    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)  # eval_set naming changed across lightgbm versions
        model.fit(
            train[FEATURES],
            train[TARGET],
            eval_set=[(valid[FEATURES], valid[TARGET])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    return model


# --------------------------------------------------------------------------- metrics


def topk_metrics(y: np.ndarray, score: np.ndarray, share: float) -> dict:
    k = max(int(round(share * len(y))), 1)
    top = np.argsort(-score, kind="stable")[:k]
    hits = y[top].sum()
    base = y.mean()
    return {
        "precision_at_k": float(hits / k),
        "recall_at_k": float(hits / max(y.sum(), 1)),
        "lift_at_k": float((hits / k) / base) if base > 0 else float("nan"),
    }


def evaluate(y: np.ndarray, score: np.ndarray, share: float, prob: np.ndarray | None = None) -> dict:
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "base_rate": float(y.mean()),
        **topk_metrics(y, score, share),
    }
    if prob is not None:
        out["brier"] = float(brier_score_loss(y, prob))
        out["ece"] = expected_calibration_error(y, prob)
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- backtest


def split_for(test_cutoff: int, cutoffs: list[int], cfg: config.ChurnConfig) -> tuple[list[int], list[int]]:
    """Only cutoffs whose labels were complete by the test week can be used."""
    available = [c for c in cutoffs if c + cfg.horizon <= test_cutoff]
    cal = available[-cfg.calibration_cutoffs :]
    train = available[: -cfg.calibration_cutoffs]
    return train, cal


def backtest(frame: pd.DataFrame, cfg: config.ChurnConfig, track: bool) -> tuple[pd.DataFrame, dict]:
    cutoffs = sorted(int(c) for c in frame.cutoff.unique())
    rows, last = [], {}
    for test_cut in cfg.test_cutoffs:
        train_cuts, cal_cuts = split_for(test_cut, cutoffs, cfg)
        tr = frame[frame.cutoff.isin(train_cuts)]
        ca = frame[frame.cutoff.isin(cal_cuts)]
        te = frame[frame.cutoff == test_cut]
        y = te[TARGET].to_numpy()

        logit = LogitModel().fit(tr, tr[TARGET].to_numpy())
        gbm = fit_lgbm(tr, ca)
        p_cal = gbm.predict_proba(ca[FEATURES])[:, 1]
        cal = PlattCalibrator().fit(p_cal, ca[TARGET].to_numpy())
        iso = IsotonicCalibrator().fit(p_cal, ca[TARGET].to_numpy())  # compared, not used
        raw = gbm.predict_proba(te[FEATURES])[:, 1]
        prob = cal(raw)

        # ranking metrics use the raw score (Platt doesn't change the order anyway);
        # calibration metrics use the calibrated probability, which is what the money maths uses
        scores = {
            "rule: recent spend drop": (heuristic_score(te), None),
            "logistic regression": (logit.predict(te), logit.predict(te)),
            "LightGBM": (raw, prob),
        }
        for name, (score, p) in scores.items():
            m = evaluate(y, score, cfg.top_k_share, p)
            if name == "LightGBM":
                m["brier_uncalibrated"] = float(brier_score_loss(y, raw))
                m["ece_uncalibrated"] = expected_calibration_error(y, raw)
                m["ece_isotonic"] = expected_calibration_error(y, iso(raw))
                m["platt_a"], m["platt_b"] = cal.a, cal.b
            rows.append(
                {
                    "test_cutoff": test_cut,
                    "model": name,
                    "train_cutoffs": f"{train_cuts[0]}-{train_cuts[-1]}",
                    "cal_cutoffs": f"{cal_cuts[0]}-{cal_cuts[-1]}",
                    "n_train": len(tr),
                    "n_test": len(te),
                    **m,
                }
            )
        with tracking.run(f"churn-backtest-{test_cut}", enabled=track, tags={"stage": "backtest"}) as ml:
            if ml:
                ml.log_params(
                    {
                        **{k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
                        "test_cutoff": test_cut,
                        "train_cutoffs": f"{train_cuts[0]}-{train_cuts[-1]}",
                        "best_iteration": gbm.best_iteration_,
                    }
                )
                for r in rows[-len(scores) :]:
                    tag = r["model"].split(" ")[0].replace(":", "").lower()
                    ml.log_metrics(
                        {f"{tag}_{k}": r[k] for k in ("pr_auc", "roc_auc", "recall_at_k", "lift_at_k")}
                    )
        last = {"test": te, "raw": raw, "prob": prob, "y": y, "test_cutoff": test_cut}
    return pd.DataFrame(rows), last


# --------------------------------------------------------------------------- final model


def train_final(
    frame: pd.DataFrame, cfg: config.ChurnConfig
) -> tuple[lgb.LGBMClassifier, PlattCalibrator, dict]:
    cutoffs = sorted(int(c) for c in frame.cutoff.unique())
    cal_cuts = cutoffs[-cfg.calibration_cutoffs :]
    train_cuts = cutoffs[: -cfg.calibration_cutoffs]
    tr, ca = frame[frame.cutoff.isin(train_cuts)], frame[frame.cutoff.isin(cal_cuts)]
    gbm = fit_lgbm(tr, ca)
    cal = PlattCalibrator().fit(gbm.predict_proba(ca[FEATURES])[:, 1], ca[TARGET].to_numpy())
    meta = {
        "train_cutoffs": train_cuts,
        "calibration_cutoffs": cal_cuts,
        "best_iteration": int(gbm.best_iteration_),
        "n_train": int(len(tr)),
        "features": FEATURES,
        "target": f"shops in <= {cfg.quiet_max} of the next {cfg.horizon} weeks",
        "population": f"shopped in >= {cfg.min_regular} of the last {cfg.lookback} weeks",
    }
    return gbm, cal, meta


def save_model(gbm: lgb.LGBMClassifier, cal: PlattCalibrator, meta: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gbm.booster_.save_model(str(config.MODEL_DIR / "lapse_lgbm.txt"), num_iteration=gbm.best_iteration_)
    (config.MODEL_DIR / "lapse_calibrator.json").write_text(json.dumps(cal.to_dict()))
    (config.MODEL_DIR / "lapse_meta.json").write_text(json.dumps(meta, indent=2))


def score_latest(con, gbm, cal) -> pd.DataFrame:
    """Score every currently active household as of the last week of data."""
    last_week = int(con.execute("SELECT max(week) FROM dim_week").fetchone()[0])
    f = features_at(con, last_week)
    f["is_regular"] = is_regular(f)
    f["lapse_prob"] = np.where(f.is_regular, cal(gbm.predict_proba(f[FEATURES])[:, 1]), np.nan)
    reg = f.is_regular
    f.loc[reg, "risk_decile"] = pd.qcut(f.loc[reg, "lapse_prob"].rank(method="first"), 10, labels=False) + 1
    return f


# --------------------------------------------------------------------------- figures


def _fig_backtest(metrics: pd.DataFrame, cfg: config.ChurnConfig) -> None:
    order = ["rule: recent spend drop", "logistic regression", "LightGBM"]
    agg = (
        metrics[metrics.model.isin(order)]
        .groupby("model")[["pr_auc", "recall_at_k"]]
        .agg(["mean", "min", "max"])
    )
    agg = agg.reindex(order)
    base = metrics.base_rate.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (col, title, sub) in zip(
        axes,
        [
            ("pr_auc", "PR-AUC (area under precision-recall)", f"random guessing = base rate ≈ {base:.2f}"),
            (
                "recall_at_k",
                f"Share of lapsers caught in the top {cfg.top_k_share:.0%}",
                f"random = {cfg.top_k_share:.0%}",
            ),
        ],
        strict=True,
    ):
        y = np.arange(len(order))[::-1]
        means = agg[(col, "mean")].to_numpy()
        colors = [P.NEUTRAL, P.NEUTRAL, P.BLUE]
        ax.barh(y, means, color=colors, height=0.5)
        ax.errorbar(
            means,
            y,
            xerr=[means - agg[(col, "min")], agg[(col, "max")] - means],
            fmt="none",
            ecolor=P.INK_2,
            lw=1,
            capsize=3,
        )
        for yi, v in zip(y, means, strict=True):
            ax.text(v + 0.01, yi + 0.3, f"{v:.2f}", fontsize=9, color=P.INK, va="center")
        ref = base if col == "pr_auc" else cfg.top_k_share
        ax.axvline(ref, color=P.INK_2, lw=0.9, ls="-")
        ax.set_yticks(y, order)
        ax.set_xlim(0, max(means) * 1.3)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=11)
        P.subtitle(ax, sub + "; whiskers = min/max over 3 backtest folds")
    fig.tight_layout()
    P.save(fig, "churn_1_backtest")


def _fig_calibration(last: dict) -> None:
    y, raw, prob = last["y"], last["raw"], last["prob"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    top = 0.0
    curves = []
    for p, name, color in [(raw, "LightGBM raw", P.ORANGE), (prob, "after Platt scaling", P.BLUE)]:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        mp = np.array([p[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()])
        my = np.array([y[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()])
        curves.append((mp, my, name, color))
        top = max(top, mp.max(), my.max())
    lim = min(1.0, top * 1.2)
    ax.plot([0, lim], [0, lim], color=P.NEUTRAL, lw=1, label="perfect calibration")
    for mp, my, name, color in curves:
        ax.plot(mp, my, color=color, label=name)
        ax.plot(mp, my, "o", color=color, mec=P.SURFACE, mew=1.5, ms=6)
    ax.set_xlabel("predicted probability of going quiet")
    ax.set_ylabel("observed rate")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(axis="x")
    ax.set_title("Calibration on the last test fold")
    P.subtitle(ax, f"deciles of predicted risk, test cutoff = week {last['test_cutoff']}")
    ax.legend(loc="upper left", fontsize=8.5)
    P.save(fig, "churn_2_calibration")


def _fig_shap(gbm, X: pd.DataFrame) -> dict:
    import shap

    explainer = shap.TreeExplainer(gbm.booster_)
    sv = explainer(X[FEATURES])
    values = sv.values if sv.values.ndim == 2 else sv.values[..., 1]
    imp = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.barh([FEATURE_DESCRIPTIONS[f] for f in top.index], top.values, color=P.BLUE, height=0.55)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP value| (log-odds of lapsing)")
    ax.set_title("What drives lapse risk")
    P.subtitle(ax, "global importance on the last test fold, top 12 features")
    P.save(fig, "churn_3_shap_importance")

    fig = plt.figure()
    shap.plots.beeswarm(
        shap.Explanation(
            values,
            base_values=sv.base_values if np.ndim(sv.base_values) == 1 else sv.base_values[..., 1],
            data=X[FEATURES].to_numpy(),
            feature_names=[FEATURE_DESCRIPTIONS[f] for f in FEATURES],
        ),
        max_display=12,
        show=False,
        plot_size=(8.4, 5.2),
    )
    plt.title("Direction of each feature's effect", loc="left", fontsize=12, fontweight="semibold", pad=14)
    P.save(plt.gcf(), "churn_4_shap_beeswarm")

    # three individual explanations: high, middle and low predicted risk
    p = gbm.predict_proba(X[FEATURES])[:, 1]
    picks = {"high": int(np.argmax(p)), "median": int(np.argsort(p)[len(p) // 2]), "low": int(np.argmin(p))}
    cases = {}
    for name, i in picks.items():
        contrib = pd.Series(values[i], index=FEATURES)
        topc = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
        cases[name] = {
            "household_id": int(X.iloc[i]["household_id"]),
            "raw_prob": float(p[i]),
            "top_contributions": [
                {"feature": f, "value": float(X.iloc[i][f]), "shap": float(v)} for f, v in topc.items()
            ],
        }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (name, case) in zip(axes, cases.items(), strict=True):
        c = pd.DataFrame(case["top_contributions"])[::-1]
        ax.barh(
            [FEATURE_DESCRIPTIONS[f] for f in c.feature],
            c.shap,
            color=[P.NEG if v > 0 else P.POS for v in c.shap],
            height=0.55,
        )
        ax.axvline(0, color=P.INK_2, lw=0.8)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{name} risk: household {case['household_id']}", fontsize=10)
        P.subtitle(ax, f"raw model probability {case['raw_prob']:.0%}")
    fig.suptitle(
        "Why the model scored these households the way it did (red pushes risk up)",
        x=0.01,
        y=1.06,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    P.save(fig, "churn_5_shap_cases")
    return {"importance": imp.round(4).to_dict(), "cases": cases}


# --------------------------------------------------------------------------- run


def run(cfg: config.ChurnConfig = config.SETTINGS.churn, track: bool = True) -> dict:
    con = connect()
    try:
        frame = training_frame(con, cfg.cutoffs())
        frame = frame[is_regular(frame)].reset_index(drop=True)
        log.info(
            "training frame: %d rows over %d cutoffs, lapse rate %.3f",
            len(frame),
            frame.cutoff.nunique(),
            frame[TARGET].mean(),
        )
        metrics, last = backtest(frame, cfg, track)
        summary = (
            metrics.groupby("model")[
                [
                    "pr_auc",
                    "roc_auc",
                    "precision_at_k",
                    "recall_at_k",
                    "lift_at_k",
                    "brier",
                    "ece",
                    "brier_uncalibrated",
                    "ece_uncalibrated",
                    "ece_isotonic",
                ]
            ]
            .mean()
            .round(4)
        )
        log.info("\n%s", summary.to_string())

        gbm, cal, meta = train_final(frame, cfg)
        save_model(gbm, cal, meta)
        scores = score_latest(con, gbm, cal)
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(config.ARTIFACTS_DIR / "lapse_scores_latest.parquet", index=False)
    finally:
        con.close()

    _fig_backtest(metrics, cfg)
    _fig_calibration(last)
    shap_info = _fig_shap(gbm, last["test"].sample(min(2000, len(last["test"])), random_state=config.SEED))

    best = summary.loc["LightGBM"]
    rule = summary.loc["rule: recent spend drop"]
    results = {
        "target": meta["target"],
        "population": meta["population"],
        "n_rows": int(len(frame)),
        "cutoffs": cfg.cutoffs(),
        "test_cutoffs": list(cfg.test_cutoffs),
        "base_rate": float(frame[TARGET].mean()),
        "full_lapse_rate": float(frame["full_lapse"].mean()),
        "spend_halved_rate": float(frame["spend_halved"].mean()),
        "folds": metrics.round(4).to_dict("records"),
        "summary": summary.reset_index().to_dict("records"),
        "final_model": meta,
        "shap": shap_info,
        "scored_households": int(scores.is_regular.sum()),
        "headline": (
            f"On 3 out-of-time test folds LightGBM reached PR-AUC {best.pr_auc:.2f} "
            f"(base rate {frame[TARGET].mean():.2f}, ROC-AUC {best.roc_auc:.2f}) and caught {best.recall_at_k:.0%} of the "
            f"regular shoppers about to go quiet in its top {cfg.top_k_share:.0%}, vs {rule.recall_at_k:.0%} for the day-one rule."
        ),
    }
    with tracking.run("churn-final", enabled=track, tags={"stage": "final"}) as ml:
        if ml:
            ml.log_params(
                {
                    "best_iteration": meta["best_iteration"],
                    "train_cutoffs": f"{meta['train_cutoffs'][0]}-{meta['train_cutoffs'][-1]}",
                }
            )
            ml.log_metrics({f"backtest_{k}": float(v) for k, v in best.items() if pd.notna(v)})
            for f in sorted(config.FIG_DIR.glob("churn_*.png")):
                ml.log_artifact(str(f), artifact_path="figures")
            ml.log_artifact(str(config.MODEL_DIR / "lapse_lgbm.txt"), artifact_path="model")
    (config.RESULTS_DIR / "churn.json").write_text(json.dumps(results, indent=2, default=float))
    log.info(results["headline"])
    return results
