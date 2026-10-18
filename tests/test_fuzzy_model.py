import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hfm import config, data, evaluation
from hfm.fuzzy_model import CLASSES, HybridFuzzyClassifier, centroid, output_membership

ATTRS = ["age", "thalch", "oldpeak", "cp", "ca", "thal"]


@pytest.fixture(scope="module")
def split():
    df = data.load_dataset()
    train, test = data.make_split(df)
    Xtr, ytr = data.split_xy(train)
    Xte, yte = data.split_xy(test)
    return Xtr, ytr, Xte, yte


@pytest.fixture(scope="module")
def model(split):
    Xtr, ytr, *_ = split
    return HybridFuzzyClassifier(attributes=ATTRS).fit(Xtr, ytr)


def test_output_sets_cover_axis_and_centroid_is_monotone():
    u = np.linspace(0, 1, 201)
    mu = np.stack([output_membership(u, c) for c in CLASSES])
    assert np.allclose(mu.sum(axis=0), 1.0)              # Ruspini partition: memberships sum to 1 at every u
    pure = np.eye(3)
    r = centroid(pure, u, mu)
    assert r[0] < r[1] < r[2] and 0 <= r[0] and r[2] <= 1
    assert centroid(np.array([[0.0, 0.0, 0.0]]), u, mu)[0] == 0.5   # degenerate: midpoint


def test_rules_are_distributions_with_support(model):
    rt = model.rule_table()
    assert set(rt["attribute"]) == set(ATTRS)
    assert np.allclose(rt[CLASSES].sum(axis=1), 1.0)
    assert (rt["support"] > 0).all()
    assert len(rt) == 3 * 3 + 4 + 4 + 3                 # 3 continuous x 3 terms + cp + ca + thal


def test_weights_and_cutpoints(model):
    w = model.weights_
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all()
    assert model.attribute_skill_.loc["ca", "n_available"] < model.attribute_skill_.loc["age", "n_available"]
    t1, t2 = model.cutpoints_
    assert 0 < t1 < t2 < 1


def test_predict_and_scores(model, split):
    Xtr, ytr, Xte, yte = split
    risk = model.risk_score(Xte)
    assert risk.shape == (len(Xte),) and (risk >= 0).all() and (risk <= 1).all()
    pred = model.predict(Xte)
    assert set(pred) <= set(CLASSES)
    proba = model.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert evaluation.score(yte, pred)["macro_f1"] > 0.4


def test_missing_attributes_are_skipped_and_weights_renormalised(model, split):
    Xtr, *_ = split
    row = Xtr.iloc[[0]].copy()
    row[["thalch", "oldpeak", "cp", "ca", "thal"]] = np.nan      # only age measured
    ex = model.explain(row)
    assert ex["n_attributes_used"] == 1
    age = next(r for r in ex["attributes"] if r["attribute"] == "age")
    assert not age["missing"] and age["effective_weight"] == pytest.approx(1.0)
    assert np.allclose(list(ex["combined_scores"].values()), list(age["scores"].values()), atol=1e-3)
    others = [r for r in ex["attributes"] if r["attribute"] != "age"]
    assert all(r["missing"] and r["effective_weight"] == 0 and r["fired_terms"] == {} for r in others)
    # Nothing measured at all: uniform scores, centroid at the middle, still a valid class.
    row[:] = np.nan
    ex = model.explain(row)
    assert ex["n_attributes_used"] == 0 and ex["risk_score"] == pytest.approx(0.5, abs=0.05)
    assert ex["predicted"] in CLASSES


def test_explain_is_consistent_with_predict(model, split):
    Xtr, ytr, Xte, yte = split
    row = Xte.iloc[[5]]
    ex = model.explain(row)
    assert ex["predicted"] == model.predict(row)[0]
    assert ex["risk_score"] == pytest.approx(model.risk_score(row)[0])
    total = sum(np.array(list(r["contribution"].values())) for r in ex["attributes"] if not r["missing"])
    assert np.allclose(total, list(ex["combined_scores"].values()), atol=1e-3)
    assert np.isclose(sum(r["effective_weight"] for r in ex["attributes"]), 1.0)


def test_sklearn_clone_and_cross_validate(split):
    Xtr, ytr, *_ = split
    est = HybridFuzzyClassifier(attributes=ATTRS, n_inner=2)
    clone(est)
    res = evaluation.cross_validate("fuzzy", est, Xtr, ytr, cv=evaluation.make_cv(2))
    assert len(res.fold_metrics) == 2 and res.summary["macro_f1_mean"] > 0.4


def test_argmax_decision_variant(split):
    Xtr, ytr, Xte, yte = split
    m = HybridFuzzyClassifier(attributes=ATTRS, decision="argmax").fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert (pred == np.array(CLASSES)[np.argmax(m.predict_proba(Xte), axis=1)]).all()
    with pytest.raises(ValueError):
        HybridFuzzyClassifier(attributes=ATTRS, decision="nope").fit(Xtr, ytr).predict(Xte)
