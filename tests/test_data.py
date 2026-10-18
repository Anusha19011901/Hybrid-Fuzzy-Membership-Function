import numpy as np
import pandas as pd
import pytest

from hfm import config, data


@pytest.fixture(scope="module")
def df():
    return data.load_dataset()


def test_shape_and_columns(df):
    assert len(df) == 920
    assert config.TARGET in df.columns
    assert set(config.FEATURE_COLS).issubset(df.columns)


def test_zero_vitals_become_missing(df):
    for col in config.ZERO_MEANS_MISSING:
        assert not (df[col] == 0).any()
    # 172 zero-cholesterol rows plus 30 originally blank
    assert df["chol"].isna().sum() == 202


def test_risk_mapping_counts(df):
    counts = df[config.TARGET].value_counts()
    assert counts["Low"] == 411
    assert counts["Moderate"] == 265
    assert counts["High"] == 244


def test_alternative_mapping():
    raw = data.load_raw()
    risk = data.map_risk(raw[config.RAW_TARGET], "severe_only")
    counts = pd.Series(risk).value_counts()
    assert counts["High"] == 135 and counts["Moderate"] == 374


def test_split_is_stratified_and_seeded(df):
    a_tr, a_te = data.make_split(df)
    b_tr, b_te = data.make_split(df)
    assert len(a_te) == 184
    assert a_tr["id"].tolist() == b_tr["id"].tolist()
    assert set(a_tr["id"]).isdisjoint(a_te["id"])
    full = df[config.TARGET].value_counts(normalize=True)
    test = a_te[config.TARGET].value_counts(normalize=True)
    assert np.allclose(full[config.RISK_LEVELS], test[config.RISK_LEVELS], atol=0.01)
