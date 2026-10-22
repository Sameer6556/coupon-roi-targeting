"""Has the household population drifted since the model was trained?

Population Stability Index per feature, comparing the earliest training cutoff with the
week the model scores. Rule of thumb: < 0.1 stable, 0.1-0.25 keep an eye on it,
> 0.25 the model is being asked about households it never saw. Seasonality (the
holidays) is the obvious suspect in a one-year dataset.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from . import plotting as P
from .features import FEATURE_DESCRIPTIONS, FEATURES, features_at, is_regular
from .warehouse import connect

log = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI = sum (cur% - ref%) x ln(cur% / ref%) over quantile bins of the reference."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature: compare category shares instead
        values = np.union1d(np.unique(reference), np.unique(current))
        ref = np.array([(reference == v).mean() for v in values])
        cur = np.array([(current == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, edges)[0] / len(reference)
        cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def run() -> dict:
    meta = json.loads((config.MODEL_DIR / "lapse_meta.json").read_text())
    ref_cut = int(meta["train_cutoffs"][0])
    con = connect()
    try:
        cur_cut = int(con.execute("SELECT max(week) FROM dim_week").fetchone()[0])
        ref = features_at(con, ref_cut)
        cur = features_at(con, cur_cut)
    finally:
        con.close()
    ref, cur = ref[is_regular(ref)], cur[is_regular(cur)]
    rows = [{"feature": f, "psi": psi(ref[f].to_numpy(float), cur[f].to_numpy(float)),
             "ref_mean": float(ref[f].mean()), "cur_mean": float(cur[f].mean())} for f in FEATURES]
    table = pd.DataFrame(rows).sort_values("psi", ascending=False)
    table["status"] = pd.cut(table.psi, [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "watch", "drifted"]).astype(str)

    top = table.head(12)[::-1]
    colors = [P.NEG if s == "drifted" else (P.YELLOW if s == "watch" else P.NEUTRAL) for s in top.status]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([FEATURE_DESCRIPTIONS[f] for f in top.feature], top.psi, color=colors, height=0.55)
    for x, label in [(0.1, "watch"), (0.25, "drifted")]:
        ax.axvline(x, color=P.INK_2, lw=0.8)
        ax.text(x, len(top) - 0.4, f" {label}", fontsize=8, color=P.INK_2)
    for y, (v, s) in enumerate(zip(top.psi, top.status, strict=True)):
        ax.text(v + 0.005, y, f"{v:.2f} {s}", va="center", fontsize=8, color=P.INK_2)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("population stability index")
    ax.set_title("Feature drift: first training week vs scoring week")
    P.subtitle(ax, f"regular shoppers, week {ref_cut} vs week {cur_cut}; colour and label both mark the status")
    P.save(fig, "drift_1_psi")

    results = {
        "reference_cutoff": ref_cut,
        "current_cutoff": cur_cut,
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "table": table.round(4).to_dict("records"),
        "drifted": table.loc[table.status == "drifted", "feature"].tolist(),
        "watch": table.loc[table.status == "watch", "feature"].tolist(),
    }
    worst = table.iloc[0]
    results["headline"] = (
        f"{len(results['drifted'])} features drifted (PSI > 0.25) and {len(results['watch'])} are worth watching between "
        f"week {ref_cut} and week {cur_cut}; the biggest shift is {FEATURE_DESCRIPTIONS[worst.feature]} (PSI {worst.psi:.2f})."
    )
    (config.RESULTS_DIR / "drift.json").write_text(json.dumps(results, indent=2, default=float))
    log.info(results["headline"])
    return results
