"""Turn scores into a decision: who gets the next booklet, and how many should we send?

Expected incremental profit of mailing household i:

    profit_i = uplift_i ($/week) x weeks x gross margin  -  cost per booklet

Mail everyone with profit_i > 0, best first. Four ways to rank households are compared:
the shrunk uplift model, the retailer's apparent current rule (heaviest shoppers),
highest lapse risk, and random. Margins and costs are assumptions (the data has
neither), so every money figure is labelled simulated and a sensitivity grid varies them.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from . import plotting as P

log = logging.getLogger(__name__)
POLICIES = {
    "uplift model": "uplift",
    "heaviest shoppers (current rule)": "spend_16w",
    "highest lapse risk": "lapse_rank_score",
    "random": "random_score",
}
COLORS = {
    "uplift model": P.BLUE,
    "heaviest shoppers (current rule)": P.ORANGE,
    "highest lapse risk": P.AQUA,
    "random": P.NEUTRAL,
}


def load_scored() -> pd.DataFrame:
    up = pd.read_parquet(config.ARTIFACTS_DIR / "uplift_scores_latest.parquet")
    lapse = pd.read_parquet(config.ARTIFACTS_DIR / "lapse_scores_latest.parquet")[["household_id", "lapse_prob", "is_regular"]]
    df = up.merge(lapse, on="household_id", how="left")
    # non-regulars have no lapse score; they go to the back of the lapse-risk queue
    df["lapse_rank_score"] = df["lapse_prob"].fillna(-1.0)
    df["random_score"] = np.random.default_rng(config.SEED).random(len(df))
    return df


def expected_profit(uplift: np.ndarray, weeks: float, margin: float, cost: float) -> np.ndarray:
    return uplift * weeks * margin - cost


def profit_curve(df: pd.DataFrame, score: str, profit: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-df[score].to_numpy(), kind="stable")
    cum = np.cumsum(profit[order])
    n = len(df)
    return pd.DataFrame({"mailed": np.arange(1, n + 1), "share": np.arange(1, n + 1) / n, "cum_profit": cum})


def holdout_curves(pairs: pd.DataFrame, model_col: str, weeks: float, margin: float, cost: float) -> dict:
    """Realised profit on held-out matched pairs (waves the models never saw)."""
    out = {}
    for name, col in [("uplift model", model_col), ("heaviest shoppers (current rule)", "heaviest shoppers"), ("random", "random")]:
        order = np.argsort(-pairs[col].to_numpy(), kind="stable")
        d = pairs["delta"].to_numpy()[order]
        prof = d * weeks * margin - cost
        out[name] = pd.DataFrame({"share": np.arange(1, len(d) + 1) / len(d), "profit_per_1000": 1000 * np.cumsum(prof) / len(d)})
    return out


def sensitivity(df: pd.DataFrame, weeks: float, biz: config.BusinessConfig) -> pd.DataFrame:
    rows = []
    for m in biz.margin_grid:
        for c in biz.cost_grid:
            prof = expected_profit(df["uplift"].to_numpy(), weeks, m, c)
            pos = prof > 0
            rows.append(
                {
                    "margin": m,
                    "cost": c,
                    "share_to_mail": float(pos.mean()),
                    "profit_per_1000_eligible": float(1000 * prof[pos].sum() / len(df)),
                    "profit_if_mail_all_per_1000": float(1000 * prof.sum() / len(df)),
                }
            )
    return pd.DataFrame(rows)


def run(biz: config.BusinessConfig = config.SETTINGS.business) -> dict:
    df = load_scored()
    causal = json.loads((config.RESULTS_DIR / "causal.json").read_text())
    uplift_res = json.loads((config.RESULTS_DIR / "uplift.json").read_text())
    weeks = float(causal["pooled"]["post_weeks_avg"])
    ate = causal["pooled"]["did_matched"]
    profit = expected_profit(df["uplift"].to_numpy(), weeks, biz.gross_margin, biz.cost_per_contact)
    df["expected_profit"] = profit
    n = len(df)

    curves = {name: profit_curve(df, col, profit) for name, col in POLICIES.items()}
    budget_share = 0.45  # Type A waves reached ~1,100 of ~2,450 active households
    k_budget = int(budget_share * n)
    at_budget = {name: float(c.cum_profit.iloc[k_budget - 1]) for name, c in curves.items()}
    best_k = int((profit > 0).sum())
    mail_all = float(profit.sum())

    # uncertainty: shift every household's uplift by the ends of the ATE confidence interval
    band = {}
    for label, shift in [("lo", ate["lo"] - ate["est"]), ("hi", ate["hi"] - ate["est"])]:
        p2 = expected_profit(df["uplift"].to_numpy() + shift, weeks, biz.gross_margin, biz.cost_per_contact)
        band[label] = profit_curve(df, "uplift", p2)

    pairs = pd.read_parquet(config.ARTIFACTS_DIR / "uplift_oot_pairs.parquet")
    hold = holdout_curves(pairs, uplift_res["chosen_learner"], weeks, biz.gross_margin, biz.cost_per_contact)
    sens = sensitivity(df, weeks, biz)

    mailing = df.loc[df.expected_profit > 0, ["household_id", "uplift", "expected_profit", "lapse_prob", "spend_16w"]]
    mailing = mailing.sort_values("expected_profit", ascending=False)
    mailing.to_csv(config.RESULTS_DIR / "next_wave_mailing_list.csv", index=False)
    df.to_parquet(config.ARTIFACTS_DIR / "policy_scored.parquet", index=False)

    _fig_expected(curves, band, n, k_budget, best_k)
    _fig_holdout(hold)
    _fig_sensitivity(sens, biz)

    results = {
        "assumptions": {"gross_margin": biz.gross_margin, "cost_per_booklet": biz.cost_per_contact, "campaign_weeks": weeks,
                        "note": "simulated: the data has no margins or costs"},
        "eligible_households": n,
        "budget_share_compared": budget_share,
        "profit_at_budget": at_budget,
        "profit_mail_everyone": mail_all,
        "optimal_mail_count": best_k,
        "optimal_mail_share": best_k / n,
        "profit_at_optimum": float(curves["uplift model"].cum_profit.iloc[best_k - 1]) if best_k else 0.0,
        "profit_at_optimum_ci": [
            float(band["lo"].cum_profit.iloc[best_k - 1]) if best_k else 0.0,
            float(band["hi"].cum_profit.iloc[best_k - 1]) if best_k else 0.0,
        ],
        "holdout_profit_per_1000_at_30pct": {k: float(v.loc[(v.share - 0.3).abs().idxmin(), "profit_per_1000"]) for k, v in hold.items()},
        "holdout_profit_per_1000_all": {k: float(v.profit_per_1000.iloc[-1]) for k, v in hold.items()},
        "sensitivity": sens.round(4).to_dict("records"),
    }
    best = max(at_budget, key=at_budget.get)
    results["headline"] = (
        f"Simulated: at the current budget ({budget_share:.0%} of active households) the uplift ranking is worth "
        f"{P.money(at_budget['uplift model'])} vs {P.money(at_budget['heaviest shoppers (current rule)'])} for the heaviest-shopper rule "
        f"and {P.money(at_budget['random'])} at random. Mailing only the {best_k:,} households with positive expected profit "
        f"({best_k / n:.0%}) gives {P.money(results['profit_at_optimum'])} "
        f"(range {P.money(results['profit_at_optimum_ci'][0])} to {P.money(results['profit_at_optimum_ci'][1])} across the effect's 95% CI). "
        f"Best ranking at budget: {best}."
    )
    (config.RESULTS_DIR / "policy.json").write_text(json.dumps(results, indent=2, default=float))
    log.info(results["headline"])
    return results


# --------------------------------------------------------------------------- figures


def _fig_expected(curves: dict, band: dict, n: int, k_budget: int, best_k: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(band["lo"].share, band["lo"].cum_profit, band["hi"].cum_profit, color=P.BLUE, alpha=0.08, lw=0)
    for name, c in curves.items():
        ax.plot(c.share, c.cum_profit, color=COLORS[name], lw=2.2 if name == "uplift model" else 1.6, label=name)
    ax.axhline(0, color=P.INK_2, lw=0.9)
    ax.axvline(k_budget / n, color=P.MUTED, lw=0.8)
    ax.text(k_budget / n, ax.get_ylim()[1] * 0.97, " current budget", fontsize=8.5, color=P.INK_2, va="top")
    if best_k:
        y = curves["uplift model"].cum_profit.iloc[best_k - 1]
        ax.plot(best_k / n, y, "o", color=P.BLUE, mec=P.SURFACE, mew=2, ms=9)
        ax.annotate(f"stop here: {best_k:,} booklets\n{P.money(y)} simulated profit", (best_k / n, y), xytext=(12, 10),
                    textcoords="offset points", fontsize=8.5, color=P.INK)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: P.money(v)))
    ax.set_xlabel("share of active households mailed, best-ranked first")
    ax.set_ylabel("cumulative expected profit (simulated)")
    ax.set_title("Next wave: how many booklets, and to whom?")
    P.subtitle(ax, "expected profit = shrunk uplift x weeks x margin - cost; band = effect's 95% CI")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(axis="x")
    P.save(fig, "policy_1_expected_profit")


def _fig_holdout(hold: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for name, c in hold.items():
        ax.plot(c.share, c.profit_per_1000, color=COLORS[name], lw=2.2 if name == "uplift model" else 1.6, label=name)
    ax.axhline(0, color=P.INK_2, lw=0.9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: P.money(v)))
    ax.set_xlabel("share of recipients kept on the list, ranked by each score")
    ax.set_ylabel("realised profit per 1,000 households")
    ax.set_title("Reality check on held-out waves (noisy)")
    P.subtitle(ax, "same profit formula, but using what matched pairs actually spent; waves the models never saw")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x")
    P.save(fig, "policy_2_holdout_profit")


def _fig_sensitivity(sens: pd.DataFrame, biz: config.BusinessConfig) -> None:
    piv = sens.pivot(index="margin", columns="cost", values="profit_per_1000_eligible").sort_index(ascending=False)
    share = sens.pivot(index="margin", columns="cost", values="share_to_mail").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    vmax = float(np.abs(piv.to_numpy()).max()) or 1.0
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", [P.NEG, P.MID, P.POS])
    ax.imshow(piv.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, s = piv.iat[i, j], share.iat[i, j]
            ax.text(j, i - 0.12, P.money(v), ha="center", va="center", fontsize=9, color=P.INK)
            ax.text(j, i + 0.22, f"mail {s:.0%}", ha="center", va="center", fontsize=7.5, color=P.INK_2)
    ax.set_xticks(range(piv.shape[1]), [f"${c:.2f}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{m:.0%}" for m in piv.index])
    ax.set_xlabel("cost per booklet")
    ax.set_ylabel("gross margin")
    ax.grid(False)
    ax.set_title("Sensitivity: profit per 1,000 households at the best list size")
    P.subtitle(ax, f"simulated; base case = {biz.gross_margin:.0%} margin, ${biz.cost_per_contact:.2f} per booklet")
    P.save(fig, "policy_3_sensitivity")
