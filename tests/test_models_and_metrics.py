import numpy as np
import pandas as pd
from scipy import stats

from couponroi.churn import expected_calibration_error, topk_metrics
from couponroi.drift import psi
from couponroi.features import FEATURES
from couponroi.models import PlattCalibrator, TLearner, XLearner, shrink
from couponroi.policy import expected_profit, profit_curve


def test_topk_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    m = topk_metrics(y, score, share=0.3)  # top 3 = ids 0, 1, 2 -> 2 hits
    assert m["precision_at_k"] == 2 / 3
    assert m["recall_at_k"] == 2 / 3
    assert np.isclose(m["lift_at_k"], (2 / 3) / 0.3)


def test_ece_is_small_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p * 1.8, 0, 1)) > 0.1


def test_platt_calibrator_fixes_a_biased_model_and_keeps_order():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.01, 0.4, 5000)
    y = (rng.random(5000) < true_p).astype(int)
    biased = np.clip(true_p * 2, 0, 0.99)
    cal = PlattCalibrator().fit(biased, y)
    fixed = cal(biased)
    assert abs(fixed.mean() - y.mean()) < 0.01
    assert (np.diff(fixed[np.argsort(biased)]) >= -1e-12).all()  # monotone: ranking unchanged
    assert np.allclose(PlattCalibrator.from_dict(cal.to_dict())(biased), fixed)  # survives a save/load


def test_psi():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(1.0, 1, 5000)) > 0.25


def _uplift_data(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    tau = 3 * (X["redemptions_16w"] > 0) - 1 * (X["spend_slope_rel_16w"])
    t = rng.integers(0, 2, n)
    y = X["spend_16w"] * 2 + t * tau + rng.normal(0, 1, n)
    return X, t, y.to_numpy(), tau.to_numpy()


def test_uplift_learners_find_planted_heterogeneity():
    X, t, y, tau = _uplift_data()
    for cls in (TLearner, XLearner):
        pred = cls().fit(X.iloc[:3000], t[:3000], y[:3000]).predict(X.iloc[3000:])
        assert stats.spearmanr(pred, tau[3000:]).statistic > 0.6, cls.__name__


def test_shrink():
    raw = np.array([0.0, 2.0, 4.0])
    assert np.allclose(shrink(raw, ate=1.0, lam=0.0, center=2.0), 1.0)
    assert np.allclose(shrink(raw, ate=1.0, lam=1.0, center=2.0), [-1.0, 1.0, 3.0])


def test_profit_maths_and_curve():
    p = expected_profit(np.array([1.0, 0.0]), weeks=7, margin=0.25, cost=1.0)
    assert np.allclose(p, [0.75, -1.0])
    df = pd.DataFrame({"score": [0.1, 0.9, 0.5]})
    c = profit_curve(df, "score", np.array([1.0, 3.0, -2.0]))
    assert c.cum_profit.tolist() == [3.0, 1.0, 2.0]  # ranked 0.9, 0.5, 0.1
