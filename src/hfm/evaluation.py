"""Stratified cross-validation, held-out scoring, and per-hospital diagnostics.

Every model in this project, baseline or fuzzy, is scored through the same
two functions so that the comparison tables line up:

* :func:`cross_validate` refits the whole pipeline (preprocessing included)
  inside each fold and returns per-fold metrics plus out-of-fold predictions.
* :func:`score` computes the metric set for one (y_true, y_pred) pair.

Per-hospital breakdowns are a *diagnostic*: the hospital is never a feature,
but because each site recorded different attributes and has a different
case-mix, a model can look fine overall while failing one site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from hfm import config

METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def score(y_true, y_pred, labels: list[str] | None = None) -> dict[str, float]:
    """Accuracy, macro F1 and weighted F1 over the given label set."""
    labels = config.RISK_LEVELS if labels is None else labels
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def per_class_f1(y_true, y_pred) -> pd.Series:
    vals = f1_score(y_true, y_pred, labels=config.RISK_LEVELS, average=None, zero_division=0)
    return pd.Series(vals, index=config.RISK_LEVELS, name="f1")


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=config.RISK_LEVELS)
    return pd.DataFrame(cm, index=pd.Index(config.RISK_LEVELS, name="true"),
                        columns=pd.Index(config.RISK_LEVELS, name="predicted"))


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=config.RISK_LEVELS, zero_division=0)


@dataclass
class CVResult:
    """Everything one cross-validated model run produced."""

    name: str
    fold_metrics: pd.DataFrame            # one row per fold
    oof_pred: np.ndarray                  # out-of-fold prediction for every training row
    fitted: list[BaseEstimator] = field(default_factory=list)

    @property
    def summary(self) -> pd.Series:
        s = self.fold_metrics[METRICS].agg(["mean", "std"]).T
        return pd.Series({f"{m}_{k}": s.loc[m, k] for m in METRICS for k in ("mean", "std")},
                         name=self.name)


def make_cv(n_splits: int = config.CV_FOLDS, seed: int = config.SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
    keep_models: bool = False,
) -> CVResult:
    """Refit ``estimator`` in every fold of a stratified K-fold on (X, y).

    ``estimator`` should be a full pipeline (preprocessing + model) so that
    imputation means, clip fences and scalers are learned inside the fold.
    """
    cv = make_cv() if cv is None else cv
    y = np.asarray(y)
    oof = np.empty(len(y), dtype=object)
    rows = []
    fitted = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        est = clone(estimator).fit(X.iloc[tr], y[tr])
        pred = est.predict(X.iloc[va])
        oof[va] = pred
        rows.append({"fold": k, **score(y[va], pred)})
        if keep_models:
            fitted.append(est)
    return CVResult(name, pd.DataFrame(rows), oof, fitted)


def per_site(y_true, y_pred, site) -> pd.DataFrame:
    """Metrics per source hospital, as a diagnostic.

    Macro F1 is computed over the classes actually present at that site
    (Hungary has no High-risk patients), so absent classes do not drag the
    average to zero. Accuracy and weighted F1 are unaffected by this choice.
    """
    y_true, y_pred, site = (np.asarray(a) for a in (y_true, y_pred, site))
    rows = []
    for s in sorted(np.unique(site)):
        m = site == s
        present = [c for c in config.RISK_LEVELS if c in set(y_true[m])]
        rows.append({"site": s, "n": int(m.sum()), **score(y_true[m], y_pred[m], labels=present)})
    return pd.DataFrame(rows).set_index("site")


def summarize(results: list[CVResult]) -> pd.DataFrame:
    """One row per model: mean and std of each CV metric."""
    return pd.DataFrame([r.summary for r in results])
