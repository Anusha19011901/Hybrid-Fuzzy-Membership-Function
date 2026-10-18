import json

import numpy as np
import pandas as pd
import pytest

from hfm import config, data
from hfm.membership import (ContinuousMembership, CrispMembership, Fuzzifier, HybridTerm,
                            count_peaks, gaussian, shape_check, trapezoid)


@pytest.fixture(scope="module")
def train():
    df = data.load_dataset()
    tr, _ = data.make_split(df)
    return tr


def test_trapezoid_shape_and_open_shoulders():
    x = np.array([0, 1, 2, 3, 4, 5])
    assert trapezoid(x, 1, 2, 3, 4).tolist() == [0, 0, 1, 1, 0, 0]
    assert trapezoid(np.array([1.5, 3.5]), 1, 2, 3, 4).tolist() == [0.5, 0.5]
    inf = float("inf")
    assert trapezoid(np.array([-100, 2, 3.5, 5]), -inf, -inf, 3, 4).tolist() == [1, 1, 0.5, 0]
    assert trapezoid(np.array([0, 2.5, 3, 100]), 2, 3, inf, inf).tolist() == [0, 0.5, 1, 1]


def test_hybrid_is_elementwise_max():
    t = HybridTerm("T", 1, 2, 3, 4, mu=2.5, sigma=1.0)
    x = np.linspace(-2, 7, 200)
    assert np.allclose(t(x), np.maximum(t.trapezoid(x), t.gaussian(x)))
    assert t(x).max() == 1.0 and t(x).min() >= 0
    assert gaussian(np.array([2.5]), 2.5, 1.0)[0] == 1.0


def test_continuous_fit_uses_observed_values_only():
    rng = np.random.default_rng(0)
    v = pd.Series(rng.normal(50, 10, 500))
    with_nan = v.copy(); with_nan.iloc[::3] = np.nan
    a = ContinuousMembership.fit(v.where(with_nan.notna()), "age")
    b = ContinuousMembership.fit(with_nan, "age")
    assert a.params().equals(b.params())
    assert a.train_mean == pytest.approx(with_nan.mean())
    # Missing values get zero membership everywhere.
    m = a.membership(np.array([np.nan, 50.0]))
    assert m.iloc[0].sum() == 0 and m.iloc[1].max() > 0
    assert a.term_names == ["Low", "Normal", "High"]
    low, normal, high = a.terms.values()
    assert np.isinf(low.a) and np.isinf(high.d)
    assert low.mu < normal.mu < high.mu
    assert low.c <= normal.b and normal.c <= high.b


def test_degenerate_quantiles_get_minimum_width():
    v = pd.Series([0.0] * 300 + list(np.linspace(0, 6, 100)))   # oldpeak-like pile-up at 0
    fam = ContinuousMembership.fit(v, "oldpeak")
    for t in fam.terms.values():
        if not np.isinf(t.a):
            assert t.b - t.a > 0
        if not np.isinf(t.d):
            assert t.d - t.c > 0
    assert shape_check(fam)["quasi_concave"].all()


def test_crisp_membership_one_hot():
    fam = CrispMembership.fit(pd.Series(["b", "a", "b", None, "c"]), "cp")
    assert fam.categories == ["a", "b", "c"] and fam.train_mode == "b"
    m = fam.membership(np.array(["a", "c", "zzz", None], dtype=object))
    assert m.values.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    ca = CrispMembership.fit(pd.Series([0.0, 1.0, 0.0, 3.0, np.nan]), "ca")
    assert ca.categories == ["0", "1", "3"] and ca.train_mode == "0"
    assert ca.membership(np.array([np.nan, 1.0, 3.0])).values.tolist() == [[0, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_fuzzifier_end_to_end(train):
    attrs = ["age", "oldpeak", "cp", "ca", "exang"]
    fz = Fuzzifier(attrs).fit(train)
    M = fz.transform(train)
    assert M.shape[0] == len(train)
    assert list(M.columns[:3]) == ["age=Low", "age=Normal", "age=High"]
    assert ((M >= 0) & (M <= 1)).all().all()
    assert np.allclose(M[fz.term_columns("cp")].sum(axis=1), 1.0)     # cp never missing: one category
    assert (M[fz.term_columns("age")].max(axis=1) > 0).all()          # age never missing
    # A missing reading gives zero membership in every term of that attribute, and is recorded.
    for a in ["ca", "exang", "oldpeak"]:
        miss = train[a].isna()
        assert fz.last_missing_[a].sum() == miss.sum() > 0
        assert (M.loc[miss, fz.term_columns(a)] == 0).all().all()
        assert np.allclose(M.loc[~miss, fz.term_columns(a)].max(axis=1) > 0, True)
    # Parameters come from observed rows only: refitting each attribute on its
    # own observed rows gives identical parameters.
    for a in ["age", "oldpeak"]:
        obs = train.dropna(subset=[a])
        assert fz.families[a].params().equals(Fuzzifier([a]).fit(obs).params())


def test_count_peaks_detects_w_shape():
    x = np.linspace(0, 10, 500)
    single = np.exp(-0.5 * ((x - 5) / 1) ** 2)
    plateau = trapezoid(x, 2, 4, 6, 8)
    w = np.maximum(trapezoid(x, 1, 2, 3, 4), trapezoid(x, 6, 7, 8, 9))  # two bumps
    assert count_peaks(single) == 1 and count_peaks(plateau) == 1 and count_peaks(w) == 2
    assert count_peaks(np.ones(10)) == 1
    assert count_peaks(1 - x / 10) == 1


def test_shape_check_flags_detached_gaussian():
    fam = ContinuousMembership.fit(pd.Series(np.linspace(0, 100, 300)), "age")
    bad = fam.terms["Low"]
    bad.mu, bad.sigma = 90.0, 3.0            # Gaussian far outside the Low plateau
    chk = shape_check(fam).set_index("term")
    assert not chk.loc["Low", "quasi_concave"] and chk.loc["Low", "peaks"] == 2
    assert chk.loc["Normal", "quasi_concave"] and chk.loc["High", "quasi_concave"]


def test_gaussian_mean_is_clamped_to_plateau_and_terms_stay_single_peaked(train):
    fz = Fuzzifier(config.CONTINUOUS_COLS).fit(train)
    for a in config.CONTINUOUS_COLS:
        fam = fz.families[a]
        assert shape_check(fam)["quasi_concave"].all()
        for t in fam.terms.values():
            lo_b = fam.lo if np.isinf(t.b) else t.b
            hi_c = fam.hi if np.isinf(t.c) else t.c
            assert lo_b <= t.mu <= hi_c
    # The skewed oldpeak "Normal" core mean lies outside its plateau and is the case the clamp exists for.
    p = fz.families["oldpeak"].params().set_index("term")
    assert p.loc["Normal", "mu_clamped"] and p.loc["Normal", "mu_raw"] > p.loc["Normal", "c"]
