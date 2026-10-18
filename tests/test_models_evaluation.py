import numpy as np
import pandas as pd
import pytest

from hfm import config, data, evaluation
from hfm.models import make_baselines, make_pipeline


@pytest.fixture(scope="module")
def split():
    df = data.load_dataset()
    train, test = data.make_split(df)
    Xtr, ytr = data.split_xy(train)
    Xte, yte = data.split_xy(test)
    return Xtr, ytr, Xte, yte, train, test


def test_score_metrics_in_unit_interval():
    y = ["Low", "High", "Moderate", "Low"]
    s = evaluation.score(y, y)
    assert s == {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}
    s = evaluation.score(y, ["High"] * 4)
    assert 0 <= s["macro_f1"] < s["accuracy"] <= 1


def test_confusion_frame_layout():
    cm = evaluation.confusion_frame(["Low", "High"], ["Low", "Moderate"])
    assert list(cm.index) == config.RISK_LEVELS and list(cm.columns) == config.RISK_LEVELS
    assert cm.loc["High", "Moderate"] == 1 and cm.values.sum() == 2


def test_per_site_uses_present_classes_only():
    y_true = np.array(["Low", "Moderate", "Low", "Moderate", "High", "Low"])
    y_pred = np.array(["Low", "Moderate", "Moderate", "Moderate", "High", "Low"])
    site = np.array(["A", "A", "A", "A", "B", "B"])
    out = evaluation.per_site(y_true, y_pred, site)
    assert out.loc["A", "n"] == 4 and out.loc["B", "n"] == 2
    # Site A has no High patients; macro F1 must not be dragged down by that class.
    assert out.loc["A", "macro_f1"] > 0.6
    assert out.loc["B", "accuracy"] == 1.0


def test_cross_validate_refits_preprocessing_per_fold(split):
    Xtr, ytr, *_ = split
    pipe = make_pipeline(make_baselines()["logistic_regression"])
    res = evaluation.cross_validate("lr", pipe, Xtr, ytr, cv=evaluation.make_cv(3), keep_models=True)
    assert len(res.fold_metrics) == 3
    assert set(res.oof_pred) <= set(config.RISK_LEVELS)
    assert len(res.oof_pred) == len(ytr)
    means = [m.named_steps["pre"].named_transformers_["cont"].named_steps["impute"].statistics_[2]
             for m in res.fitted]
    assert len(set(np.round(means, 6))) == 3  # different folds, different cholesterol means
    assert res.summary["macro_f1_mean"] > 0.5


def test_majority_baseline_matches_class_prior(split):
    Xtr, ytr, Xte, yte, *_ = split
    pipe = make_pipeline(make_baselines()["majority_class"]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    assert set(pred) == {"Low"}
    assert np.isclose(evaluation.score(yte, pred)["accuracy"], (yte == "Low").mean())


def test_mlp_baseline_is_seeded_and_early_stopping(split):
    Xtr, ytr, Xte, yte, *_ = split
    a = make_pipeline(make_baselines()["mlp"]).fit(Xtr, ytr)
    b = make_pipeline(make_baselines()["mlp"]).fit(Xtr, ytr)
    assert (a.predict(Xte) == b.predict(Xte)).all()
    mlp = a.named_steps["model"]
    assert mlp.hidden_layer_sizes == (32, 16) and mlp.early_stopping
    assert mlp.n_iter_ < mlp.max_iter


def test_pipeline_attribute_subset(split):
    Xtr, ytr, Xte, yte, *_ = split
    pipe = make_pipeline(make_baselines()["logistic_regression"], attributes=["age", "cp", "thal"]).fit(Xtr, ytr)
    cols = list(pipe.named_steps["pre"].get_feature_names_out())
    assert cols == ["age", "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
                    "thal_fixed defect", "thal_normal", "thal_reversable defect"]
    assert set(pipe.predict(Xte)) <= set(config.RISK_LEVELS)
    # categorical-only subset: no numeric branch at all
    pipe = make_pipeline(make_baselines()["logistic_regression"], attributes=["sex", "exang"]).fit(Xtr, ytr)
    assert list(pipe.named_steps["pre"].get_feature_names_out()) == ["sex_Male", "exang_True"]
