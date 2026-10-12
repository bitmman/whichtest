"""DeLong implementation validated against sklearn's AUC and a paired bootstrap."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from whichtest.delong import delong_auc_variance, delong_ci, delong_paired


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    n = 300
    y = (rng.random(n) < 0.35).astype(int)
    a = y * 1.2 + rng.normal(size=n)
    b = y * 0.6 + rng.normal(size=n)
    return y, a, b


def test_auc_matches_sklearn(data):
    y, a, b = data
    auc, var = delong_auc_variance(y, a)
    assert auc == pytest.approx(roc_auc_score(y, a), rel=1e-12)
    assert var > 0


def test_auc_with_ties_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 5, 200).astype(float)  # heavy ties
    auc, _ = delong_auc_variance(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), rel=1e-12)


def test_variance_matches_bootstrap(data):
    y, a, _ = data
    _, var = delong_auc_variance(y, a)
    rng = np.random.default_rng(0)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(3000):
        i = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        boots.append(roc_auc_score(y[i], a[i]))
    assert np.sqrt(var) == pytest.approx(np.std(boots, ddof=1), rel=0.15)


def test_paired_difference_matches_bootstrap(data):
    y, a, b = data
    res = delong_paired(y, a, b)
    assert res["diff"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b), rel=1e-12)
    rng = np.random.default_rng(0)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(3000):
        i = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        boots.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    assert res["se"] == pytest.approx(np.std(boots, ddof=1), rel=0.15)
    assert 0 < res["p"] < 1


def test_identical_scores_give_zero_difference(data):
    y, a, _ = data
    res = delong_paired(y, a, a.copy())
    assert res["diff"] == 0 and res["p"] == 1.0


def test_ci_is_clipped_and_ordered(data):
    y, a, _ = data
    auc, (lo, hi), se = delong_ci(y, a)
    assert 0 <= lo < auc < hi <= 1


def test_rejects_single_class():
    with pytest.raises(ValueError):
        delong_auc_variance(np.ones(10, int), np.arange(10.0))
