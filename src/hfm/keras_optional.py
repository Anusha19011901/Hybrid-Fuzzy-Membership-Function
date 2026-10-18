"""Optional Keras MLP baseline. Not used by the default pipeline or tests.

The patent compared the fuzzy model against a Keras Sequential network. This
module keeps an equivalent scikit-learn-compatible wrapper for anyone who
wants to reproduce that exactly. It is *not* imported anywhere by default:
on macOS, TensorFlow and scikit-learn each load their own OpenMP runtime and
the first Keras training step can deadlock after scikit-learn has trained a
model in the same process. The default neural-network baseline is therefore
``sklearn.neural_network.MLPClassifier`` (see :mod:`hfm.models`).

Usage::

    from hfm.keras_optional import KerasMLPClassifier   # import before sklearn fits
    make_pipeline(KerasMLPClassifier()).fit(X_train, y_train)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from hfm import config


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """A small fully-connected network with a scikit-learn interface.

    Two hidden layers with ReLU and dropout, softmax output, Adam, early
    stopping on an internal validation split. TensorFlow is imported lazily
    so the rest of the package works without it.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 32,
        patience: int = 25,
        validation_split: float = 0.15,
        random_state: int = config.SEED,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state

    def _build(self, n_in: int, n_out: int):
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.utils.set_random_seed(self.random_state)
        layers = [keras.Input(shape=(n_in,))]
        for width in self.hidden:
            layers += [keras.layers.Dense(width, activation="relu"), keras.layers.Dropout(self.dropout)]
        layers.append(keras.layers.Dense(n_out, activation="softmax"))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        from tensorflow import keras

        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.model_ = self._build(X.shape[1], len(self.classes_))
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        hist = self.model_.fit(
            X, y_idx,
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[stop],
            verbose=0,
            shuffle=True,
        )
        self.n_epochs_ = len(hist.history["loss"])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
