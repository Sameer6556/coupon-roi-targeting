import numpy as np
import pandas as pd

from couponroi import config
from couponroi.causal import analyze_wave, greedy_match, list_waves, smd, stacked_effect
from couponroi.synthetic import TRUE_EFFECT


def test_matching_respects_caliper_strata_and_no_reuse():
    rng = np.random.default_rng(0)
    n = 400
    treated = (rng.random(n) < 0.4).astype(int)
    logit = rng.normal(size=n) + treated
    strata = rng.integers(0, 2, n)
    pairs = greedy_match(treated, logit, caliper_sd=0.2, strata=strata)
    assert pairs.c.is_unique  # without replacement
    assert (treated[pairs.t] == 1).all() and (treated[pairs.c] == 0).all()
    assert (strata[pairs.t] == strata[pairs.c]).all()
    assert (np.abs(logit[pairs.t] - logit[pairs.c]) <= 0.2 * logit.std() + 1e-12).all()


def test_smd_is_zero_when_groups_are_identical():
    df = pd.DataFrame({"treated": [1, 1, 0, 0], "x": [1.0, 2.0, 1.0, 2.0]})
    pairs = pd.DataFrame({"t": [0, 1], "c": [2, 3]})
    out = smd(df, pairs, ["x"])
    assert abs(out.before.iloc[0]) < 1e-12 and abs(out.after.iloc[0]) < 1e-12


def test_waves_have_clean_baselines(con):
    waves = list_waves(con)
    assert len(waves) == 4
    for prev, cur in zip(waves, waves[1:], strict=False):
        assert cur.pre_start > prev.end  # baseline never overlaps the previous wave
        assert cur.pre_weeks >= config.SETTINGS.causal.min_pre_weeks


def test_matched_did_recovers_the_planted_effect_and_beats_naive(con):
    """The synthetic data targets heavier shoppers and adds TRUE_EFFECT x (1 or 2) $/week."""
    frames, naive = [], []
    for w in list_waves(con):
        res, m = analyze_wave(con, w)
        frames.append(m)
        naive.append(res["naive_post"]["est"])
        assert res["max_abs_smd_after"] < res["max_abs_smd_before"]
    est = stacked_effect(pd.concat(frames, ignore_index=True))
    assert est["lo"] < TRUE_EFFECT * 1.3 < est["hi"] or abs(est["est"] - TRUE_EFFECT * 1.3) < 2.0
    assert np.mean(naive) > est["est"] + 5  # the naive gap is mostly targeting bias
