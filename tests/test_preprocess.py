import numpy as np
import pandas as pd

from hfm import config, data
from hfm.preprocess import (IQRClipper, build_preprocessor, feature_groups,
                            fit_transform_split, fitted_numeric_stats,
                            columns_for_attributes, to_clinical_units)


def _split():
    df = data.load_dataset()
    train, test = data.make_split(df)
    Xtr, _ = data.split_xy(train)
    Xte, _ = data.split_xy(test)
    return Xtr, Xte


def test_iqr_clipper_bounds():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    clip = IQRClipper(k=1.5).fit(X)
    out = clip.transform(np.array([[-50.0], [1.5], [500.0]]))
    assert out[0, 0] == clip.lower_[0]
    assert out[1, 0] == 1.5
    assert out[2, 0] == clip.upper_[0]


def test_no_missing_after_transform():
    Xtr, Xte = _split()
    A, B, _ = fit_transform_split(Xtr, Xte)
    assert not A.isna().any().any()
    assert not B.isna().any().any()
    assert list(A.columns) == list(B.columns)


def test_statistics_come_from_train_only():
    Xtr, Xte = _split()
    pre = build_preprocessor(scaler="none").fit(Xtr)
    stats = fitted_numeric_stats(pre)
    # Clipping happens before imputation, so the mean is of the clipped observed values.
    clipped = Xtr[config.CONTINUOUS_COLS].clip(stats.loc[config.CONTINUOUS_COLS, "clip_lower"],
                                               stats.loc[config.CONTINUOUS_COLS, "clip_upper"], axis=1)
    assert np.allclose(stats.loc[config.CONTINUOUS_COLS, "mean"], clipped.mean())
    assert np.isclose(stats.loc["ca", "mean"], Xtr["ca"].mean())
    # Refitting on a different subset must give different statistics.
    other = build_preprocessor(scaler="none").fit(Xtr.iloc[:200])
    assert not np.allclose(fitted_numeric_stats(other)["mean"], stats["mean"])


def test_discrete_count_survives_clipping():
    """Regression: ca is 66% missing; fences must not collapse it to its mean."""
    Xtr, Xte = _split()
    A, _, _ = fit_transform_split(Xtr, Xte, scaler="none")
    assert A["ca"].nunique() >= 4
    assert A["ca"].max() == 3


def test_clip_fences_use_observed_values_only():
    Xtr, Xte = _split()
    pre = build_preprocessor(scaler="none").fit(Xtr)
    stats = fitted_numeric_stats(pre)
    q1, q3 = Xtr["chol"].quantile([0.25, 0.75])
    assert np.isclose(stats.loc["chol", "clip_lower"], q1 - 1.5 * (q3 - q1))
    assert np.isclose(stats.loc["chol", "clip_upper"], q3 + 1.5 * (q3 - q1))


def test_standard_scaling_is_train_centred():
    Xtr, Xte = _split()
    A, B, _ = fit_transform_split(Xtr, Xte, scaler="standard")
    assert np.allclose(A[config.NUMERIC_COLS].mean(), 0, atol=1e-9)
    # Test split is not exactly centred: proof that it was only transformed.
    assert np.abs(B[config.NUMERIC_COLS].mean()).max() > 1e-3


def test_minmax_and_none_scalers():
    Xtr, Xte = _split()
    A, _, _ = fit_transform_split(Xtr, Xte, scaler="minmax")
    assert A[config.NUMERIC_COLS].min().min() >= 0 and A[config.NUMERIC_COLS].max().max() <= 1
    C, _, _ = fit_transform_split(Xtr, Xte, scaler="none")
    assert C["age"].max() > 20  # still in years


def test_feature_groups_cover_every_attribute():
    Xtr, Xte = _split()
    _, _, pre = fit_transform_split(Xtr, Xte)
    groups = feature_groups(pre)
    assert set(groups) == set(config.FEATURE_COLS)
    assert groups["sex"] == ["sex_Male"]
    assert len(groups["cp"]) == 4


def test_columns_for_attributes_keeps_onehot_groups_whole():
    Xtr, Xte = _split()
    _, _, pre = fit_transform_split(Xtr, Xte)
    cols = columns_for_attributes(pre, ["age", "cp", "exang"])
    assert cols == ["age", "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal",
                    "cp_typical angina", "exang_True"]
    import pytest
    with pytest.raises(KeyError):
        columns_for_attributes(pre, ["cp_asymptomatic"])


def test_to_clinical_units_roundtrip():
    Xtr, Xte = _split()
    for scaler in ("standard", "minmax", "none"):
        A, B, pre = fit_transform_split(Xtr, Xte, scaler=scaler)
        back = to_clinical_units(pre, B)
        assert list(back.columns) == config.FEATURE_COLS
        # Rows where nothing was missing or clipped must come back exactly.
        stats = fitted_numeric_stats(pre)
        ok = Xte[config.NUMERIC_COLS].notna().all(axis=1)
        for c in config.CONTINUOUS_COLS:
            ok &= Xte[c].between(stats.loc[c, "clip_lower"], stats.loc[c, "clip_upper"])
        ok &= Xte[config.CATEGORICAL_COLS].notna().all(axis=1)
        assert ok.sum() > 50
        assert np.allclose(back.loc[ok, config.NUMERIC_COLS], Xte.loc[ok, config.NUMERIC_COLS])
        assert (back.loc[ok, config.CATEGORICAL_COLS] == Xte.loc[ok, config.CATEGORICAL_COLS]).all().all()
        assert back["age"].max() > 20 and back["chol"].max() > 100
