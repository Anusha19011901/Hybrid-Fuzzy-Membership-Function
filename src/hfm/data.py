"""Loading, cleaning and labelling the UCI Heart Disease dataset.

The pipeline is deliberately split into small pure functions so each decision
(what counts as missing, how `num` becomes a risk label, how the split is
drawn) is visible and testable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hfm import config


def load_raw(path: Path | str = config.RAW_CSV) -> pd.DataFrame:
    """Read the Kaggle CSV exactly as shipped (920 rows, 16 columns)."""
    df = pd.read_csv(path)
    expected = {config.ID_COL, config.SITE_COL, config.RAW_TARGET, *config.FEATURE_COLS}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV layout, missing columns: {sorted(missing)}")
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the row-independent fixes that do not need a fitted statistic.

    * Physiologically impossible zeros in `trestbps` / `chol` become NaN so
      that the (train-fitted) mean imputer treats them as missing rather than
      dragging the mean towards zero.
    * Boolean-ish columns (`fbs`, `exang`) are coerced to the strings
      "True"/"False" so that pandas does not carry a mixed bool/NaN object
      column into the one-hot encoder.
    * `ca` is kept numeric; its integer range 0-3 is small but ordered.

    Nothing here depends on the training split, so it is safe to run on the
    full table before splitting.
    """
    out = df.copy()
    for col in config.ZERO_MEANS_MISSING:
        out.loc[out[col] == 0, col] = np.nan
    for col in config.CATEGORICAL_COLS:
        out[col] = out[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype("object")
    return out


def map_risk(num: pd.Series, mapping: str = config.DEFAULT_RISK_MAPPING) -> pd.Series:
    """Collapse the 0-4 `num` target into an ordered Low/Moderate/High label."""
    table = config.RISK_MAPPINGS[mapping]
    unknown = set(num.unique()).difference(table)
    if unknown:
        raise ValueError(f"`num` contains values outside the mapping: {sorted(unknown)}")
    return pd.Categorical(num.map(table), categories=config.RISK_LEVELS, ordered=True)


def load_dataset(
    path: Path | str = config.RAW_CSV,
    mapping: str = config.DEFAULT_RISK_MAPPING,
) -> pd.DataFrame:
    """Load, clean, and attach the `risk` label. Keeps `id`, `dataset`, `num`."""
    df = clean_raw(load_raw(path))
    df[config.TARGET] = map_risk(df[config.RAW_TARGET], mapping)
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the risk label. Returns (train, test).

    The split is drawn *before* any imputation, clipping or scaling so that
    every fitted statistic in `hfm.preprocess` sees training rows only.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[config.TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model features from the risk label."""
    return df[config.FEATURE_COLS].copy(), df[config.TARGET].astype(str)
