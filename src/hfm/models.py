"""Baseline classifiers the fuzzy model is compared against.

Three real baselines (logistic regression, random forest, a small
multilayer perceptron) and one reference baseline that predicts the majority
class. The reference gives the floor for this label setup, which every other
number in the project is read against.

The patent's own neural baseline was a Keras Sequential model. Here the
network is scikit-learn's ``MLPClassifier`` with the same shape (two hidden
layers, 32 and 16 units, ReLU, Adam, early stopping). A Keras version lives
in :mod:`hfm.keras_optional` but is not part of the default pipeline because
of a TensorFlow / scikit-learn OpenMP deadlock on macOS.

Hyper-parameters are modest, fixed and untuned. The baselines exist to put
the fuzzy model in context, not to be the best possible classifiers.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from hfm import config
from hfm.preprocess import build_preprocessor


def make_baselines(seed: int = config.SEED) -> dict[str, BaseEstimator]:
    """Name -> unfitted estimator (model only, no preprocessing)."""
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        ),
    }


def make_pipeline(model: BaseEstimator, scaler: str = "standard") -> Pipeline:
    """Preprocessing + model in one estimator, so CV refits the statistics per fold."""
    return Pipeline([("pre", build_preprocessor(scaler=scaler)), ("model", model)])
