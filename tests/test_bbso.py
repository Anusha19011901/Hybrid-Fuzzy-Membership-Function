import numpy as np
import pytest

from hfm import config, data
from hfm.bbso import BBSO, BBSOConfig, MaskFitness, consensus_subset, selection_frequency, sigmoid


def _target_fitness(target):
    """Toy objective: fraction of bits that match a hidden target mask."""
    target = np.asarray(target, dtype=bool)
    return lambda mask: float((np.asarray(mask, dtype=bool) == target).mean())


def test_sigmoid_bounds():
    assert sigmoid(np.array([-50.0, 0.0, 50.0])).tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_bbso_finds_planted_optimum():
    target = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    res = BBSO(_target_fitness(target), 13, BBSOConfig(n_pop=20, max_iter=60, patience=15), seed=0).run()
    assert res.best_fitness == 1.0
    assert (res.best_mask == target).all()
    assert len(res.history) == res.n_iter
    assert np.all(np.diff(res.history) >= 0)  # best-so-far never decreases


def test_bbso_is_reproducible_per_seed():
    f = _target_fitness([1, 1, 0, 0, 1, 0, 1, 0])
    a = BBSO(f, 8, BBSOConfig(max_iter=10, patience=3), seed=7).run()
    b = BBSO(f, 8, BBSOConfig(max_iter=10, patience=3), seed=7).run()
    c = BBSO(f, 8, BBSOConfig(max_iter=10, patience=3), seed=8).run()
    assert a.history == b.history and (a.best_mask == b.best_mask).all()
    assert a.history != c.history or (a.best_mask != c.best_mask).any()


def test_bbso_early_stops_and_never_returns_empty_mask():
    res = BBSO(lambda m: 0.5, 6, BBSOConfig(max_iter=50, patience=4), seed=1).run()  # flat objective
    assert res.stopped_early and res.n_iter == 4
    assert res.best_mask.any()
    for mask, _ in res.population:
        assert mask.any()


def test_mask_fitness_caches_and_scores_reasonably():
    df = data.load_dataset()
    train, _ = data.make_split(df)
    X, y = data.split_xy(train)
    fit = MaskFitness(X, y, n_splits=3)
    assert fit.n_features == 13
    all_on = np.ones(13, dtype=bool)
    s1 = fit(all_on)
    s2 = fit(all_on)
    assert s1 == s2 and fit.n_evals == 1 and 0.5 < s1 < 0.8
    assert fit(np.zeros(13, dtype=bool)) == 0.0
    only_age = np.zeros(13, dtype=bool); only_age[0] = True
    assert fit(only_age) < s1
    assert fit.selected(only_age) == ["age"]


def test_consensus_and_frequency():
    from hfm.bbso import BBSOResult
    attrs = list("abcd")
    mk = lambda bits, f, s: BBSOResult(np.array(bits, bool), f, [f], 1, False, s)
    res = [mk([1, 1, 0, 0], 0.6, 0), mk([1, 1, 0, 0], 0.6, 1), mk([1, 0, 1, 0], 0.7, 2)]
    freq = selection_frequency(res, attrs)
    assert freq["a"] == 1.0 and freq["d"] == 0.0 and freq["b"] == pytest.approx(2 / 3)
    subset, how = consensus_subset(res, attrs)
    assert subset == ["a", "b"] and how.startswith("modal")
    res = [mk([1, 1, 0, 0], 0.6, 0), mk([1, 0, 1, 0], 0.7, 1), mk([1, 0, 0, 1], 0.5, 2)]
    subset, how = consensus_subset(res, attrs)
    assert subset == ["a"] and how.startswith("attributes selected")
