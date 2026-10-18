"""Hybrid trapezoidal + Gaussian membership functions, fitted from data.

The patent's core idea is a *hybrid* membership function: for each
linguistic term the membership of a value x is the element-wise maximum of
a trapezoid and a Gaussian evaluated at x. The trapezoid gives a bounded
plateau that reads as "clearly Low / Normal / High"; the Gaussian adds a
smooth, focused peak. The patent hand-tuned the parameters; here every
parameter comes from the training split, using observed (non-imputed)
values only:

* trapezoid break points are training quantiles (``config.TRAPEZOID_QUANTILES``),
  with the outer terms open-ended (``Low`` is fully true below its plateau,
  ``High`` above its);
* the Gaussian of a term has ``mu`` and ``sigma`` equal to the mean and
  standard deviation of the training values whose trapezoid membership in
  that term is at least 0.5, with ``mu`` clamped into the term's plateau
  ``[b, c]``. The clamp is what keeps every hybrid term single-peaked: left
  of the plateau both components are non-decreasing, right of it both are
  non-increasing, so their maximum is too. Without it a skewed attribute
  (``oldpeak``) put a Gaussian peak just outside the plateau and produced a
  second, tiny bump. ``mu_raw`` in the parameter table shows when the clamp
  acted.

Only the continuous attributes get fuzzy terms. Categorical attributes and
the vessel count ``ca`` get *crisp* memberships: one term per category,
membership 1 for the patient's own category and 0 for every other. A chest
pain type or an ECG finding is a recorded fact with no order and no
overlap, and ``ca`` takes four integer values, mostly imputed; inventing
gradations between them would add fuzziness the data does not contain.

Missing values are **not** filled for the fuzzy model. A missing reading
gets membership 0 in every term of its attribute, so no rule can fire on
it and the prediction rests on measured attributes only (the inference
step skips such attributes and renormalises the remaining weights). This
is a deliberate choice: filling with the training mean or mode would let
the fill value fire rules, and for ``ca``, ``thal`` and ``slope`` the fill
value is mostly a proxy for the source hospital. The fuzzifier records
which cells were missing so explanations can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hfm import config

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Elementary shapes
# --------------------------------------------------------------------------- #
def trapezoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership: 0 at a, rises to 1 at b, stays 1 to c, falls to 0 at d.

    ``a == b`` or ``c == d`` give vertical edges; ``-inf``/``inf`` give open shoulders.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    rise = (x >= a) & (x < b)
    out[rise] = (x[rise] - a) / max(b - a, EPS)
    out[(x >= b) & (x <= c)] = 1.0
    fall = (x > c) & (x <= d)
    out[fall] = (d - x[fall]) / max(d - c, EPS)
    if np.isinf(a):
        out[x < b] = 1.0
    if np.isinf(d):
        out[x > c] = 1.0
    return out


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mu) / max(sigma, EPS)) ** 2)


@dataclass
class HybridTerm:
    """One linguistic term: trapezoid (a, b, c, d) and Gaussian (mu, sigma)."""

    name: str
    a: float
    b: float
    c: float
    d: float
    mu: float
    sigma: float
    mu_raw: float = float("nan")   # core mean before clamping into [b, c]

    def trapezoid(self, x) -> np.ndarray:
        return trapezoid(x, self.a, self.b, self.c, self.d)

    def gaussian(self, x) -> np.ndarray:
        return gaussian(x, self.mu, self.sigma)

    def __call__(self, x) -> np.ndarray:
        """Hybrid membership: element-wise max of the trapezoid and the Gaussian."""
        return np.maximum(self.trapezoid(x), self.gaussian(x))


# --------------------------------------------------------------------------- #
# Per-attribute membership families
# --------------------------------------------------------------------------- #
@dataclass
class ContinuousMembership:
    """Three hybrid terms over one continuous attribute, in clinical units."""

    attribute: str
    terms: dict[str, HybridTerm]
    lo: float               # observed training minimum
    hi: float               # observed training maximum
    train_mean: float       # mean of observed training values (for reference only)

    @property
    def term_names(self) -> list[str]:
        return list(self.terms)

    def membership(self, x) -> pd.DataFrame:
        """Hybrid membership per term; a missing value gets 0 in every term."""
        x = np.asarray(x, dtype=float)
        ok = ~np.isnan(x)
        out = {}
        for t, term in self.terms.items():
            m = np.zeros(len(x))
            m[ok] = term(x[ok])
            out[t] = m
        return pd.DataFrame(out)

    def grid(self, n: int = 400) -> np.ndarray:
        pad = 0.05 * (self.hi - self.lo)
        return np.linspace(self.lo - pad, self.hi + pad, n)

    @classmethod
    def fit(cls, values: pd.Series, attribute: str,
            quantiles: tuple[float, ...] = config.TRAPEZOID_QUANTILES,
            min_width_frac: float = 0.05) -> "ContinuousMembership":
        """Derive the three terms from observed training values only."""
        v = pd.Series(values).dropna().astype(float).to_numpy()
        if len(v) < 10:
            raise ValueError(f"{attribute}: need at least 10 observed values, got {len(v)}")
        lo, hi = float(v.min()), float(v.max())
        q = [float(np.quantile(v, p)) for p in quantiles]
        # Guard against degenerate break points (e.g. many identical values such
        # as oldpeak == 0): keep every ramp at least a small fraction of the range.
        min_w = min_width_frac * (hi - lo)
        for i in range(1, len(q)):
            q[i] = max(q[i], q[i - 1] + min_w)
        inf = float("inf")
        shapes = {
            "Low": (-inf, -inf, q[0], q[1]),
            "Normal": (q[0], q[1], q[2], q[3]),
            "High": (q[2], q[3], inf, inf),
        }
        terms = {}
        for name, (a, b, c, d) in shapes.items():
            core = v[trapezoid(v, a, b, c, d) >= 0.5]
            mu_raw = float(core.mean())
            mu = float(np.clip(mu_raw, max(b, lo), min(c, hi)))   # keep the peak on the plateau
            sigma = float(max(core.std(ddof=0), min_w / 2))
            terms[name] = HybridTerm(name, a, b, c, d, mu, sigma, mu_raw)
        return cls(attribute, terms, lo, hi, float(v.mean()))

    def params(self) -> pd.DataFrame:
        rows = [{"attribute": self.attribute, "term": t.name, "a": t.a, "b": t.b, "c": t.c,
                 "d": t.d, "mu": t.mu, "sigma": t.sigma, "mu_raw": t.mu_raw,
                 "mu_clamped": not np.isclose(t.mu, t.mu_raw)} for t in self.terms.values()]
        return pd.DataFrame(rows)


@dataclass
class CrispMembership:
    """One crisp term per category: 1 for the patient's own category, else 0."""

    attribute: str
    categories: list[str]
    train_mode: str         # most frequent training category (for reference only)

    @property
    def term_names(self) -> list[str]:
        return list(self.categories)

    @staticmethod
    def _as_labels(x, attribute: str) -> np.ndarray:
        """Categories as strings; missing stays missing (matches no term)."""
        s = pd.Series(x)
        if attribute in config.DISCRETE_COLS:
            s = pd.to_numeric(s, errors="coerce").round()
            return s.map(lambda v: None if pd.isna(v) else str(int(v))).to_numpy(dtype=object)
        return s.map(lambda v: None if pd.isna(v) else str(v)).to_numpy(dtype=object)

    def membership(self, x) -> pd.DataFrame:
        """Crisp membership: 1 for the patient's own category, 0 otherwise, 0 everywhere if missing."""
        labels = self._as_labels(x, self.attribute)
        return pd.DataFrame({c: (labels == c).astype(float) for c in self.categories})

    @classmethod
    def fit(cls, values: pd.Series, attribute: str) -> "CrispMembership":
        v = pd.Series(values).dropna()
        if attribute in config.DISCRETE_COLS:          # ca: integer count as a category
            v = v.astype(float).round().astype(int).astype(str)
            cats = sorted(v.unique(), key=int)
        else:
            v = v.astype(str)
            cats = sorted(v.unique())
        return cls(attribute, list(cats), str(v.mode().iloc[0]))


# --------------------------------------------------------------------------- #
# The fuzzifier: raw clinical frame -> membership degrees
# --------------------------------------------------------------------------- #
@dataclass
class Fuzzifier:
    """Fit membership families on the training split, fuzzify any frame.

    ``transform`` accepts the *cleaned raw* frame (NaNs allowed, clinical
    units) and returns one column per (attribute, term) named
    ``"age=Low"``, ``"cp=asymptomatic"`` and so on. A missing cell gives 0
    in every term of its attribute. ``transform`` also records
    ``self.last_missing_``: a boolean frame marking which cells were missing.
    """

    attributes: list[str]
    families: dict[str, ContinuousMembership | CrispMembership] = field(default_factory=dict)
    last_missing_: pd.DataFrame | None = None

    @property
    def continuous(self) -> list[str]:
        return [a for a in self.attributes if a in config.CONTINUOUS_COLS]

    @property
    def crisp(self) -> list[str]:
        return [a for a in self.attributes if a not in config.CONTINUOUS_COLS]

    def fit(self, X: pd.DataFrame) -> "Fuzzifier":
        for a in self.attributes:
            if a in config.CONTINUOUS_COLS:
                self.families[a] = ContinuousMembership.fit(X[a], a)
            else:
                self.families[a] = CrispMembership.fit(X[a], a)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.last_missing_ = X[self.attributes].isna()
        blocks = []
        for a, fam in self.families.items():
            m = fam.membership(X[a].to_numpy())
            m.columns = [f"{a}={t}" for t in m.columns]
            m.index = X.index
            blocks.append(m)
        return pd.concat(blocks, axis=1)

    def term_columns(self, attribute: str) -> list[str]:
        return [f"{attribute}={t}" for t in self.families[attribute].term_names]

    def params(self) -> pd.DataFrame:
        return pd.concat([self.families[a].params() for a in self.continuous], ignore_index=True)


# --------------------------------------------------------------------------- #
# Shape check: is every hybrid term still a readable, single-peaked shape?
# --------------------------------------------------------------------------- #
def count_peaks(y: np.ndarray, tol: float = 1e-6) -> int:
    """Number of local maxima (plateaus count once) of a sampled curve."""
    d = np.diff(y)
    d = np.where(np.abs(d) < tol, 0, np.sign(d))
    d = d[d != 0]                     # drop flat stretches
    if len(d) == 0:
        return 1
    peaks = int(np.sum((d[:-1] > 0) & (d[1:] < 0)))
    if d[0] < 0:                      # starts descending: a peak at the left edge
        peaks += 1
    if d[-1] > 0:                     # ends ascending: a peak at the right edge
        peaks += 1
    return max(peaks, 1)


def shape_check(fam: ContinuousMembership, n: int = 2000) -> pd.DataFrame:
    """For each term: peaks of the hybrid, and how far the Gaussian extends the trapezoid.

    A term is *quasi-concave* (single-peaked, readable) when the hybrid curve
    has exactly one peak or plateau. ``gauss_extension`` is the largest amount
    by which the Gaussian lifts the membership above the trapezoid anywhere
    on the grid: 0 means the trapezoid dominates everywhere.
    """
    x = fam.grid(n)
    rows = []
    for t in fam.terms.values():
        trap, gau, hyb = t.trapezoid(x), t.gaussian(x), t(x)
        rows.append({
            "attribute": fam.attribute, "term": t.name,
            "peaks": count_peaks(hyb),
            "quasi_concave": count_peaks(hyb) == 1,
            "gauss_extension": float(np.max(hyb - trap)),
            "mu_inside_plateau": bool(t.b <= t.mu <= t.c),
        })
    return pd.DataFrame(rows)
