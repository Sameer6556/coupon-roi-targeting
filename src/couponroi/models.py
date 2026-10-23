"""Model classes shared by training, the API and the app.

Only numpy, pandas, scikit-learn and LightGBM are imported here, so the API container
can unpickle the uplift model without installing matplotlib, statsmodels or SHAP.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .features import FEATURES

SEED = 42
LOG_FEATURES = {
    "spend_4w",
    "spend_8w",
    "spend_16w",
    "trips_4w",
    "trips_8w",
    "trips_16w",
    "basket_value_16w",
    "redemptions_16w",
    "categories_16w",
}
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


class LogitModel:
    """Logistic regression with the same preprocessing every time: log1p on skewed
    amounts and counts, median imputation, standardisation (all fitted on train only)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogitModel:
        Z = self._prep(X, fit=True)
        self.model = LogisticRegression(max_iter=5000, C=0.5).fit(Z, y)
        return self

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        Z = X[FEATURES].astype(float).copy()
        for c in LOG_FEATURES:
            Z[c] = np.log1p(Z[c].clip(lower=0))
        if fit:
            self.medians = Z.median()
        Z = Z.fillna(self.medians)
        if fit:
            self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        return ((Z - self.mu) / self.sd).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling: p_cal = sigmoid(a * logit(p) + b), fitted on later calibration weeks.

    Two numbers, smooth and monotone (ranking is unchanged). Isotonic regression was tried
    first: similar Brier score, but it overfit the sparse top end and jumped to 1.0 for
    households with only a handful of look-alikes in the calibration set.
    """

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        return self

    @classmethod
    def from_dict(cls, d: dict) -> PlattCalibrator:
        obj = cls()
        obj.a, obj.b = float(d["a"]), float(d["b"])
        return obj

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * _logit(p) + self.b)))

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b}


class IsotonicCalibrator:
    """Kept for comparison in the write-up; see PlattCalibrator for why it isn't used."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        self.x, self.y = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self.x, self.y)


def _reg(params: dict = REG_PARAMS, seed: int = SEED) -> lgb.LGBMRegressor:
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

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = SEED) -> TLearner:
        self.m1 = _fit(_reg(seed=seed), X[t == 1], y[t == 1])
        self.m0 = _fit(_reg(seed=seed), X[t == 0], y[t == 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    """Kunzel et al. 2019. Impute each side's missing outcome, model those imputed
    effects, then blend the two effect models by the propensity score."""

    name = "X-learner"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = SEED) -> XLearner:
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

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, seed: int = SEED) -> DRLearner:
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
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final.predict(X)


LEARNERS = {"T-learner": TLearner, "X-learner": XLearner, "DR-learner": DRLearner}


def shrink(raw: np.ndarray, ate: float, lam: float, center: float) -> np.ndarray:
    """Pull raw uplift predictions toward the average effect by however much they held up
    out of time (lam = calibration slope clipped to [0, 1])."""
    return ate + lam * (np.asarray(raw, dtype=float) - center)
