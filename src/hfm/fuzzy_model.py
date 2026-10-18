"""Fuzzy inference: per-attribute rule models, weighted average, centroid.

This follows the structure described in the patent:

1. **One fuzzy model per attribute.** Each linguistic term of an attribute
   is the antecedent of a single rule ``IF attribute IS term THEN risk IS
   ...``. The consequent is learned from the training split as the
   membership-weighted class distribution of the patients in that term
   (with Laplace smoothing), so a rule reads, for example,
   ``IF thalch IS Low THEN Low 0.18 / Moderate 0.33 / High 0.49``.
   For a patient, the attribute's rules fire with the patient's term
   memberships and their consequents are combined by firing strength into
   a per-attribute class-score vector.

2. **Weighted average across attributes.** The per-attribute vectors are
   averaged with weights learned on an inner stratified CV of the training
   split: ``w = max(macro F1 of the single-attribute model - majority-class
   macro F1, 0)``, i.e. how much better than guessing that attribute alone
   is. Attributes with no membership for a patient (missing reading) are
   skipped and the remaining weights renormalised.

3. **Output variable and centroid defuzzification.** The combined class
   scores clip three output sets (Low / Moderate / High) on a risk axis
   from 0 to 1; the aggregated output is defuzzified with the centroid
   (centre of gravity) method into one crisp risk score.

4. **Cut-points.** The crisp score is turned into a class with two
   thresholds chosen on the inner-CV out-of-fold scores of the training
   split to maximise macro F1.

Optional **pairwise rule layer** (``pair_top_k``): two-antecedent rules
``IF a IS t1 AND b IS t2`` over every pair of the ``pair_top_k`` attributes
with the highest single-attribute weight. A pair fires with the product of
the two memberships, is learned and weighted exactly like a single
attribute, and rules with too little support are pruned so the rule base
stays readable.

Everything the model does for one patient is exposed by :meth:`explain`.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from hfm import config
from hfm.membership import Fuzzifier, HybridTerm, trapezoid

CLASSES = list(config.RISK_LEVELS)
N_CLASSES = len(CLASSES)

# Output sets on the risk axis u in [0, 1]: Low is a left shoulder, Moderate a
# triangle centred at 0.5, High a right shoulder. They overlap so that a
# mixture of class scores produces an intermediate centroid.
OUTPUT_SETS = {
    "Low": HybridTerm("Low", -np.inf, -np.inf, 0.15, 0.5, 0.0, 1.0),
    "Moderate": HybridTerm("Moderate", 0.15, 0.5, 0.5, 0.85, 0.5, 1.0),
    "High": HybridTerm("High", 0.5, 0.85, np.inf, np.inf, 1.0, 1.0),
}


def output_membership(u: np.ndarray, name: str) -> np.ndarray:
    t = OUTPUT_SETS[name]
    return trapezoid(u, t.a, t.b, t.c, t.d)


def centroid(scores: np.ndarray, u: np.ndarray, mu_sets: np.ndarray) -> np.ndarray:
    """Centre of gravity of max_c min(score_c, mu_c(u)) for each row of ``scores``.

    ``scores`` is (n, 3), ``u`` is the grid (m,), ``mu_sets`` is (3, m).
    """
    clipped = np.minimum(scores[:, :, None], mu_sets[None, :, :])   # (n, 3, m)
    agg = clipped.max(axis=1)                                        # (n, m)
    area = agg.sum(axis=1)
    out = np.full(len(scores), 0.5)
    ok = area > 0
    out[ok] = (agg[ok] * u).sum(axis=1) / area[ok]
    return out


class HybridFuzzyClassifier(ClassifierMixin, BaseEstimator):
    """Patent-style fuzzy classifier over the cleaned raw clinical frame.

    Parameters
    ----------
    attributes : list of clinical attributes to use (continuous ones get
        hybrid terms, the others crisp terms).
    smoothing : Laplace smoothing added to each class count of a rule.
    n_inner : folds of the inner stratified CV used for weights and cut-points.
    resolution : number of points on the output risk axis.
    decision : ``"centroid"`` (the patent's path: defuzzify, then cut-points)
        or ``"argmax"`` (largest combined class score; a diagnostic that
        bypasses the output sets, used to measure what the centroid path costs).
    pair_top_k : 0 for single-antecedent rules only; k >= 2 adds two-antecedent
        rules over every pair of the k highest-weight attributes.
    pair_min_support : pairwise rules learned from fewer (membership-weighted)
        training patients than this are pruned.
    """

    def __init__(self, attributes: list[str] | None = None, smoothing: float = 1.0,
                 n_inner: int = 3, resolution: int = 201, decision: str = "centroid",
                 pair_top_k: int = 0, pair_min_support: float = 5.0,
                 random_state: int = config.SEED):
        self.attributes = attributes
        self.smoothing = smoothing
        self.n_inner = n_inner
        self.resolution = resolution
        self.decision = decision
        self.pair_top_k = pair_top_k
        self.pair_min_support = pair_min_support
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Rule learning
    # ------------------------------------------------------------------ #
    def _attrs(self) -> list[str]:
        return list(config.FEATURE_COLS if self.attributes is None else self.attributes)

    @staticmethod
    def key_name(key) -> str:
        return key if isinstance(key, str) else " × ".join(key)

    @staticmethod
    def _firing(M: pd.DataFrame, fz: Fuzzifier, key) -> tuple[np.ndarray, list[str]]:
        """Firing strengths (n, T) and term labels for a single attribute or a pair."""
        if isinstance(key, str):
            cols = fz.term_columns(key)
            return M[cols].to_numpy(), [c.split("=", 1)[1] for c in cols]
        a, b = key
        Fa, la = M[fz.term_columns(a)].to_numpy(), [c.split("=", 1)[1] for c in fz.term_columns(a)]
        Fb, lb = M[fz.term_columns(b)].to_numpy(), [c.split("=", 1)[1] for c in fz.term_columns(b)]
        firing = (Fa[:, :, None] * Fb[:, None, :]).reshape(len(M), -1)   # product t-norm
        return firing, [f"{x} & {y}" for x in la for y in lb]

    def _rule_tables(self, M: pd.DataFrame, y: np.ndarray, fz: Fuzzifier, keys) -> dict:
        """Per key: rows = (kept) rules, columns = support, class consequents, col_idx."""
        Y = np.stack([(y == c).astype(float) for c in CLASSES], axis=1)   # (n, 3)
        tables = {}
        for key in keys:
            firing, labels = self._firing(M, fz, key)
            support = firing.sum(axis=0)
            counts = firing.T @ Y + self.smoothing                         # (T, 3)
            probs = counts / counts.sum(axis=1, keepdims=True)
            tab = pd.DataFrame(probs, index=labels, columns=CLASSES)
            tab.insert(0, "support", support)
            tab["col_idx"] = np.arange(len(labels))
            if not isinstance(key, str):
                tab = tab[tab["support"] >= self.pair_min_support]
            tables[key] = tab
        return tables

    @classmethod
    def _attribute_scores(cls, M: pd.DataFrame, fz: Fuzzifier, tables: dict) -> dict[str, np.ndarray]:
        """Per key name: (n, 3) class scores, NaN rows where nothing fires."""
        out = {}
        for key, tab in tables.items():
            firing = cls._firing(M, fz, key)[0][:, tab["col_idx"].to_numpy()]
            total = firing.sum(axis=1)
            scores = np.full((len(firing), N_CLASSES), np.nan)
            ok = total > 0
            if len(tab):
                scores[ok] = (firing[ok] @ tab[CLASSES].to_numpy()) / total[ok, None]
            out[cls.key_name(key)] = scores
        return out

    @staticmethod
    def _combine(attr_scores: dict[str, np.ndarray], weights: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Weighted average over available attributes; returns (scores, n_used)."""
        names = list(attr_scores)
        S = np.stack([attr_scores[a] for a in names], axis=0)             # (A, n, 3)
        w = weights.reindex(names).fillna(0).to_numpy()[:, None, None]
        avail = ~np.isnan(S[:, :, 0])                                      # (A, n)
        W = np.where(avail, w[:, :, 0], 0.0)                               # (A, n)
        denom = W.sum(axis=0)                                              # (n,)
        num = (np.nan_to_num(S) * W[:, :, None]).sum(axis=0)               # (n, 3)
        combined = np.full((S.shape[1], N_CLASSES), 1.0 / N_CLASSES)
        ok = denom > 0
        combined[ok] = num[ok] / denom[ok, None]
        return combined, avail.sum(axis=0)

    # ------------------------------------------------------------------ #
    # sklearn API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(str)
        attrs = self._attrs()
        self.classes_ = np.array(CLASSES)
        self.u_ = np.linspace(0, 1, self.resolution)
        self.mu_sets_ = np.stack([output_membership(self.u_, c) for c in CLASSES])

        # ---- inner CV: skill of every single attribute ------------------- #
        cv = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.random_state)
        folds = list(cv.split(X, y))
        majority = CLASSES[np.argmax([(y == c).sum() for c in CLASSES])]
        floor = f1_score(y, np.full(len(y), majority), average="macro", labels=CLASSES, zero_division=0)

        def oof_scores(keys) -> dict[str, np.ndarray]:
            oof = {self.key_name(k): np.full((len(y), N_CLASSES), np.nan) for k in keys}
            for tr, va in folds:
                fz = Fuzzifier(attrs).fit(X.iloc[tr])
                tabs = self._rule_tables(fz.transform(X.iloc[tr]), y[tr], fz, keys)
                sc = self._attribute_scores(fz.transform(X.iloc[va]), fz, tabs)
                for name in oof:
                    oof[name][va] = sc[name]
            return oof

        def skill_rows(oof: dict[str, np.ndarray], n_antecedents: int) -> list[dict]:
            rows = []
            for name, sc in oof.items():
                ok = ~np.isnan(sc[:, 0])
                pred = np.array(CLASSES)[np.argmax(sc[ok], axis=1)]
                f1 = f1_score(y[ok], pred, average="macro", labels=CLASSES, zero_division=0) if ok.any() else 0.0
                rows.append({"attribute": name, "antecedents": n_antecedents, "inner_macro_f1": f1,
                             "n_available": int(ok.sum()), "weight": max(f1 - floor, 0.0)})
            return rows

        oof_single = oof_scores(attrs)
        rows = skill_rows(oof_single, 1)
        single_w = pd.Series({r["attribute"]: r["weight"] for r in rows})

        # ---- optional pairwise layer over the top-k attributes ------------ #
        pairs = []
        oof_pairs = {}
        if self.pair_top_k and self.pair_top_k >= 2:
            top = list(single_w.sort_values(ascending=False).index[: self.pair_top_k])
            pairs = [tuple(p) for p in combinations([a for a in attrs if a in top], 2)]
            oof_pairs = oof_scores(pairs)
            rows += skill_rows(oof_pairs, 2)
        self.pairs_ = pairs

        skill = pd.DataFrame(rows).set_index("attribute")
        if skill["weight"].sum() == 0:
            skill["weight"] = 1.0
        skill["weight"] /= skill["weight"].sum()
        self.attribute_skill_ = skill
        self.weights_ = skill["weight"]
        self.f1_floor_ = floor

        oof_all = {**oof_single, **oof_pairs}
        oof_scores_c, _ = self._combine(oof_all, self.weights_)
        oof_risk = centroid(oof_scores_c, self.u_, self.mu_sets_)
        self.cutpoints_ = self._best_cutpoints(oof_risk, y)
        self.oof_risk_ = oof_risk

        # ---- final rules on all training rows ----------------------------- #
        self.fuzzifier_ = Fuzzifier(attrs).fit(X)
        self.rules_ = self._rule_tables(self.fuzzifier_.transform(X), y, self.fuzzifier_, list(attrs) + pairs)
        return self

    @staticmethod
    def _best_cutpoints(risk: np.ndarray, y: np.ndarray, n_grid: int = 40) -> tuple[float, float]:
        cand = np.unique(np.quantile(risk, np.linspace(0.05, 0.95, n_grid)))
        best, best_f1 = (0.4, 0.6), -1.0
        for i, t1 in enumerate(cand):
            for t2 in cand[i + 1:]:
                pred = np.where(risk < t1, "Low", np.where(risk < t2, "Moderate", "High"))
                f1 = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
                if f1 > best_f1:
                    best, best_f1 = (float(t1), float(t2)), f1
        return best

    def _scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
        check_is_fitted(self, "rules_")
        M = self.fuzzifier_.transform(X)
        attr_scores = self._attribute_scores(M, self.fuzzifier_, self.rules_)
        combined, n_used = self._combine(attr_scores, self.weights_)
        return combined, n_used, attr_scores, M

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Crisp defuzzified risk in [0, 1] for every row."""
        combined, _, _, _ = self._scores(X)
        return centroid(combined, self.u_, self.mu_sets_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """The combined class scores (they sum to 1 by construction)."""
        return self._scores(X)[0]

    def classify_risk(self, risk: np.ndarray) -> np.ndarray:
        t1, t2 = self.cutpoints_
        return np.where(risk < t1, "Low", np.where(risk < t2, "Moderate", "High"))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.decision == "argmax":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        if self.decision != "centroid":
            raise ValueError(f"decision must be 'centroid' or 'argmax', got {self.decision!r}")
        return self.classify_risk(self.risk_score(X))

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def rule_table(self) -> pd.DataFrame:
        """All rules (single- and two-antecedent) with their learned consequents."""
        rows = []
        for key, tab in self.rules_.items():
            name = self.key_name(key)
            for term, r in tab.iterrows():
                rows.append({"attribute": name, "antecedents": 1 if isinstance(key, str) else 2, "term": term,
                             "support": r["support"], **{c: r[c] for c in CLASSES},
                             "consequent": CLASSES[int(np.argmax(r[CLASSES].to_numpy()))],
                             "weight": self.weights_[name]})
        return pd.DataFrame(rows)

    def explain(self, X_row: pd.DataFrame) -> dict:
        """Everything that produced the prediction for one patient (a 1-row frame)."""
        assert len(X_row) == 1
        combined, n_used, attr_scores, M = self._scores(X_row)
        risk = float(centroid(combined, self.u_, self.mu_sets_)[0])
        rows = []
        avail_w = sum(self.weights_[self.key_name(k)] for k in self.rules_
                      if not np.isnan(attr_scores[self.key_name(k)][0, 0]))
        for key, tab in self.rules_.items():
            name = self.key_name(key)
            firing, labels = self._firing(M, self.fuzzifier_, key)
            kept = tab["col_idx"].to_numpy()
            fired = {labels[j]: float(firing[0, j]) for j in kept if firing[0, j] > 0}
            missing = np.isnan(attr_scores[name][0, 0])
            w_eff = 0.0 if missing or avail_w == 0 else float(self.weights_[name] / avail_w)
            if isinstance(key, str):
                value = None if pd.isna(X_row[key].iloc[0]) else X_row[key].iloc[0]
            else:
                value = ", ".join("not measured" if pd.isna(X_row[a].iloc[0]) else str(X_row[a].iloc[0]) for a in key)
            rows.append({
                "attribute": name,
                "antecedents": 1 if isinstance(key, str) else 2,
                "value": value,
                "missing": bool(missing),
                "fired_terms": fired,
                "scores": None if missing else dict(zip(CLASSES, attr_scores[name][0].round(4))),
                "weight": float(self.weights_[name]),
                "effective_weight": w_eff,
                "contribution": None if missing else dict(zip(CLASSES, (w_eff * attr_scores[name][0]).round(4))),
            })
        clipped = np.minimum(combined[0][:, None], self.mu_sets_)
        return {
            "attributes": rows,
            "n_attributes_used": int(n_used[0]),
            "combined_scores": dict(zip(CLASSES, combined[0].round(4))),
            "risk_score": risk,
            "cutpoints": self.cutpoints_,
            "predicted": str(self.classify_risk(np.array([risk]))[0]),
            "output_axis": self.u_,
            "output_sets": self.mu_sets_,
            "clipped_sets": clipped,
            "aggregated": clipped.max(axis=0),
        }
