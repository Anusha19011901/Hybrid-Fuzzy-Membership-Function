"""Leakage-safe preprocessing: imputation, outlier clipping, scaling, encoding.

The patent describes three pre-processing steps:

1. missing values replaced with the mean of the attribute,
2. standardisation / normalisation,
3. removal of outliers and inconsistencies.

This module implements them as a scikit-learn ``ColumnTransformer`` so that
every statistic (means, clip bounds, scale) is estimated with ``fit`` on the
training split and merely *applied* to the test split.

Deviations from the patent text, and why:

* Outliers are **clipped** to train-derived IQR fences (winsorised) rather
  than removed. Removing rows from a held-out set changes what is being
  evaluated, and clipping keeps every patient in the test set. Fences are
  computed on the *observed* training values, before imputation; otherwise a
  heavily-imputed column (``ca`` is 66% missing) collapses onto its mean and
  the fences clip every real value away. The discrete vessel count ``ca``
  is never clipped.
* Categorical attributes have no mean. They are imputed with the most
  frequent training value and one-hot encoded. Numeric attributes use the
  mean, as described.
* Only one scaler is applied at a time (`standard`, `minmax`, or none). The
  fuzzy phases work on unscaled, imputed values so the membership functions
  stay in clinical units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from hfm import config

ScalerName = Literal["standard", "minmax", "none"]


class IQRClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorise each column to ``[Q1 - k*IQR, Q3 + k*IQR]`` learned on fit.

    Parameters
    ----------
    k : float
        Fence multiplier. 1.5 is Tukey's conventional "outlier" fence.
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = self._validate_data(X, dtype=float, force_all_finite="allow-nan")
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        X = self._validate_data(X, dtype=float, reset=False, force_all_finite="allow-nan")
        return np.clip(X, self.lower_, self.upper_)


def _scaler(name: ScalerName):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return "passthrough"
    raise ValueError(f"Unknown scaler {name!r}; expected standard | minmax | none")


def build_preprocessor(
    scaler: ScalerName = "standard",
    clip_outliers: bool = True,
    iqr_k: float = 1.5,
    continuous_cols: list[str] | None = None,
    discrete_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Create an **unfitted** preprocessing transformer.

    Call ``.fit(X_train)`` then ``.transform`` on train and test. Use
    :func:`transform_to_frame` to get the result back as a labelled DataFrame.

    Numeric branches run ``clip -> impute -> scale`` (continuous) and
    ``impute -> scale`` (discrete). ``np.clip`` leaves NaN untouched, so the
    missing cells reach the imputer intact and the fences only ever see real
    measurements.
    """
    continuous_cols = list(config.CONTINUOUS_COLS if continuous_cols is None else continuous_cols)
    discrete_cols = list(config.DISCRETE_COLS if discrete_cols is None else discrete_cols)
    categorical_cols = list(config.CATEGORICAL_COLS if categorical_cols is None else categorical_cols)

    continuous_steps = []
    if clip_outliers:
        continuous_steps.append(("clip", IQRClipper(k=iqr_k)))
    continuous_steps += [("impute", SimpleImputer(strategy="mean")), ("scale", _scaler(scaler))]

    discrete_steps = [("impute", SimpleImputer(strategy="mean")), ("scale", _scaler(scaler))]

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ]

    branches = [
        ("cont", Pipeline(continuous_steps), continuous_cols),
        ("disc", Pipeline(discrete_steps), discrete_cols),
        ("cat", Pipeline(categorical_steps), categorical_cols),
    ]
    pre = ColumnTransformer(
        transformers=[b for b in branches if b[2]],   # a subset may empty a branch
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def _branch_columns(pre: ColumnTransformer, name: str) -> list[str]:
    for n, _, cols in pre.transformers_:
        if n == name:
            return list(cols)
    return []


def fitted_numeric_stats(pre: ColumnTransformer) -> pd.DataFrame:
    """Imputation means and clip fences learned on the training split."""
    frames = []
    cont_cols = _branch_columns(pre, "cont")
    if cont_cols:
        cont = pre.named_transformers_["cont"]
        rows = pd.DataFrame({"mean": cont.named_steps["impute"].statistics_}, index=cont_cols)
        if "clip" in cont.named_steps:
            rows["clip_lower"] = cont.named_steps["clip"].lower_
            rows["clip_upper"] = cont.named_steps["clip"].upper_
        frames.append(rows)
    disc_cols = _branch_columns(pre, "disc")
    if disc_cols:
        disc = pre.named_transformers_["disc"]
        frames.append(pd.DataFrame({"mean": disc.named_steps["impute"].statistics_}, index=disc_cols))
    return pd.concat(frames)


def transform_to_frame(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor and return a DataFrame with clean names."""
    out = pre.transform(X)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=pre.get_feature_names_out(), index=X.index)
    return out


def feature_groups(pre: ColumnTransformer) -> dict[str, list[str]]:
    """Map each original attribute to the output column(s) it produced.

    One-hot encoding turns ``cp`` into ``cp_asymptomatic``, ``cp_atypical angina``
    and so on. Later phases (feature selection, fuzzy rules) reason about the
    original clinical attribute, so this lookup is needed to go back and forth.
    """
    groups: dict[str, list[str]] = {}
    for name in pre.get_feature_names_out():
        source = next(
            (c for c in config.FEATURE_COLS if name == c or name.startswith(c + "_")), name
        )
        groups.setdefault(source, []).append(name)
    return groups


def fit_transform_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: ScalerName = "standard",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit on train, transform both. Returns (Xtr, Xte, fitted_preprocessor)."""
    pre = build_preprocessor(scaler=scaler, **kwargs)
    pre.fit(X_train)
    return transform_to_frame(pre, X_train), transform_to_frame(pre, X_test), pre


# --------------------------------------------------------------------------- #
# Going back to clinical units
# --------------------------------------------------------------------------- #
def columns_for_attributes(pre: ColumnTransformer, attributes: list[str]) -> list[str]:
    """Encoded column names produced by the given clinical attributes.

    This is the hook the feature selector uses: a binary mask over the 13
    attributes is turned into the list of encoded columns to keep, so that a
    one-hot group is always kept or dropped as a whole.
    """
    groups = feature_groups(pre)
    unknown = [a for a in attributes if a not in groups]
    if unknown:
        raise KeyError(f"Unknown attributes: {unknown}")
    return [c for a in attributes for c in groups[a]]


def to_clinical_units(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Invert scaling and one-hot encoding of a processed frame.

    Numeric columns come back in their original units (years, mm Hg, mg/dl,
    bpm). Categorical groups come back as their label (the argmax of the
    one-hot block, or the dropped level for binary columns). Imputation and
    clipping are not undone: this is what the model saw, in readable units.
    """
    out = pd.DataFrame(index=X.index)
    for name in ("cont", "disc"):
        cols = _branch_columns(pre, name)
        if not cols:
            continue
        scaler = pre.named_transformers_[name].named_steps["scale"]
        vals = X[cols].to_numpy(dtype=float)
        if scaler != "passthrough":
            vals = scaler.inverse_transform(vals)
        out[cols] = vals
    cat_cols = _branch_columns(pre, "cat")
    if cat_cols:
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        out[cat_cols] = onehot.inverse_transform(X[onehot.get_feature_names_out(cat_cols)].to_numpy())
    return out[[c for c in config.FEATURE_COLS if c in out.columns]]
