"""Diagnostic and DeLong references from pair counting, not rank-based libraries."""
from math import erfc, sqrt
from statistics import mean, variance

import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, SpecError, run
from whichtest.delong import delong_auc_variance, delong_paired


def pairwise_placements(y, scores):
    positive = [s for t, s in zip(y, scores) if t == 1]
    negative = [s for t, s in zip(y, scores) if t == 0]
    kernel = [[float(a > b) + 0.5 * (a == b) for b in negative] for a in positive]
    return [mean(row) for row in kernel], [mean(col) for col in zip(*kernel)]


@pytest.mark.parametrize("scores", [[1, 2, 2, 4, 0, 2, 3, 4], [0]*8, [5, 6, 7, 8, 0, 1, 2, 3]])
def test_delong_variance_matches_pairwise_placement_oracle(scores):
    y = [1]*4 + [0]*4
    vp, vn = pairwise_placements(y, scores)
    auc, var = delong_auc_variance(np.array(y), np.array(scores))
    assert auc == pytest.approx(mean(vp), abs=1e-12)
    assert var == pytest.approx(variance(vp)/4 + variance(vn)/4, abs=1e-12)


@pytest.mark.parametrize("transform", [lambda a: a, lambda a: 7*a + 13, lambda a: np.exp(a)])
def test_paired_delong_difference_covariance_and_tail(transform):
    y = np.array([1]*4 + [0]*4)
    a = transform(np.array([1, 2, 2, 4, 0, 2, 3, 4], dtype=float))
    b = transform(np.array([2, 1, 4, 3, 0, 1, 4, 2], dtype=float))
    ap, an = pairwise_placements(y, a)
    bp, bn = pairwise_placements(y, b)
    dp, dn = [x-z for x, z in zip(ap, bp)], [x-z for x, z in zip(an, bn)]
    diff, se = mean(dp), sqrt(variance(dp)/4 + variance(dn)/4)
    assert se > 0, "oracle fixture must have estimable nonzero variance"
    r = delong_paired(y, a, b)
    assert r["diff"] == pytest.approx(diff, abs=1e-12)
    assert r["se"] == pytest.approx(se, abs=1e-12)
    assert r["p"] == pytest.approx(erfc(abs(diff/se)/sqrt(2)), abs=1e-12)
    swapped = delong_paired(y, b, a)
    assert swapped["diff"] == pytest.approx(-diff)
    assert swapped["p"] == pytest.approx(r["p"])
    assert swapped["ci"] == pytest.approx((-r["ci"][1], -r["ci"][0]))


@pytest.mark.parametrize("missing_role", ["truth", "score"])
def test_public_auc_matches_pairs_after_joint_exclusion(missing_role):
    d = pd.DataFrame({"truth": [1., 1., 1., 0., 0., 0.], "score": [1., 2., 4., 0., 2., 3.]})
    damaged = d.copy()
    damaged.loc[1, missing_role] = np.nan
    kept = damaged.dropna()
    vp, _ = pairwise_placements(kept.truth, kept.score)
    r = run(Spec(test="roc_auc", variables={"outcome": "truth", "x": "score"}, params={"positive_label": 1}), damaged)
    assert r.effect_size.value == pytest.approx(mean(vp))
    assert r.n == {"positive": 2, "negative": 3}


@pytest.mark.parametrize("prediction", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_constant_binary_prediction_is_a_valid_diagnostic_test(prediction):
    # No requirement that a diagnostic test call at least one case positive.
    r = run(Spec(test="diag_binary", variables={"outcome": "truth", "x": "pred"}),
            pd.DataFrame({"truth": [0, 1, 0, 1], "pred": prediction}))
    expected = ({"tp": 0, "fp": 0, "fn": 2, "tn": 2} if prediction[0] == 0
                else {"tp": 2, "fp": 2, "fn": 0, "tn": 0})
    assert r.extra["table"] == expected
    estimates = {e.name: e.value for e in r.estimates}
    assert estimates["sensitivity"] == prediction[0]
    assert estimates["specificity"] == 1 - prediction[0]


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_nullable_pandas_labels_preserve_joint_complete_cases(dtype):
    d = pd.DataFrame({"truth": pd.Series([0, 1, None, 1, 0], dtype=dtype),
                      "pred": pd.Series([0, 1, 1, None, 1], dtype=dtype)})
    r = run(Spec(test="diag_binary", variables={"outcome": "truth", "x": "pred"}), d)
    assert r.n == 3
    assert r.extra["table"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}


@pytest.mark.parametrize("y", [[1, 0, 0, 0], [1, 1, 1, 0]])
def test_auc_cannot_present_unestimable_variance_as_verified(y):
    d = pd.DataFrame({"truth": y, "score": [0.2, 0.8, 0.4, 0.6]})
    try:
        r = run(Spec(test="roc_auc", variables={"outcome": "truth", "x": "score"}, params={"positive_label": 1}), d)
    except SpecError:
        return  # Clear rejection is also an acceptable public contract.
    assert r.flagged, "one case in a class cannot estimate that class's DeLong variance"
    assert r.effect_size.ci is None, "[0,1] must not conceal an undefined variance"
    assert r.warnings or any(c.failed for c in r.checks)


def test_diagnostic_metrics_match_an_asymmetric_hand_tally():
    # TP=3, FN=2, FP=1, TN=4; asymmetry prevents label swaps from hiding.
    d = pd.DataFrame({"truth": [1]*5 + [0]*5, "pred": [1,1,1,0,0,1,0,0,0,0]})
    r = run(Spec(test="diag_binary", variables={"outcome": "truth", "x": "pred"}), d)
    got = {e.name: e.value for e in r.estimates}
    for name, expected in {"sensitivity": 3/5, "specificity": 4/5, "PPV": 3/4,
                           "NPV": 4/6, "accuracy": 7/10, "LR+": 3., "LR−": .5}.items():
        assert got[name] == pytest.approx(expected)
    assert r.effect_size.value == pytest.approx(.4)
