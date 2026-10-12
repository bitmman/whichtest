"""Closed-form model answers and equivalent-input invariants.

SciPy is used only for distribution CDF/quantiles, never as a reference fit.
Fixtures are local and deterministic, independent of tests/conftest.py's RNG.
"""
from math import erfc, log, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from whichtest import Spec, SpecError, run


def tail(statistic, df, alternative):
    if alternative == "greater":
        return student_t.sf(statistic, df)
    if alternative == "less":
        return student_t.cdf(statistic, df)
    return 2 * student_t.sf(abs(statistic), df)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("level", [.8, .95])
def test_one_sample_t_closed_form_and_interval(alternative, level):
    # Mean=3, sample variance=2.5, SE=sqrt(.5), n=5; mu0=1.
    se, statistic = sqrt(.5), 2 / sqrt(.5)
    r = run(Spec(test="one_sample_t", variables={"outcome": "y"}, params={"mu0": 1},
                 alternative=alternative, ci_level=level), pd.DataFrame({"y": [1,2,3,4,5]}))
    assert r.statistic == pytest.approx(statistic)
    assert r.df == 4
    assert r.p == pytest.approx(tail(statistic, 4, alternative))
    q = student_t.ppf((1+level)/2 if alternative == "two-sided" else level, 4)
    expected = (2-q*se, 2+q*se) if alternative == "two-sided" else ((2-q*se, np.inf) if alternative == "greater" else (-np.inf, 2+q*se))
    assert r.estimates[0].ci == pytest.approx(expected)
    assert r.estimates[0].ci_level == level
    assert r.cross_check.agrees


def test_welch_unequal_sizes_and_variances_from_sufficient_statistics():
    a, b = [1,2,3,4], [2,4,6,8,10]
    # means=2.5,6; sample variances=5/3,10.
    va, vb = (5/3)/4, 10/5
    se = sqrt(va + vb)
    df = (va+vb)**2 / (va**2/3 + vb**2/4)
    r = run(Spec(test="indep_t", variables={"outcome": "y", "group": "g"}),
            pd.DataFrame({"y": a+b, "g": ["A"]*4+["B"]*5}))
    assert r.statistic == pytest.approx(-3.5 / se)
    assert r.df == pytest.approx(df)
    assert r.p == pytest.approx(2*student_t.sf(3.5/se, df))
    assert r.estimates[0].value == pytest.approx(-3.5)


def test_oneway_anova_closed_form_between_and_within_sums():
    d = pd.DataFrame({"y": [1,2,3,2,3,4,4,5,6], "g": list("aaabbbccc")})
    r = run(Spec(test="oneway_anova", variables={"outcome": "y", "group": "g"}), d)
    assert r.statistic == pytest.approx(7)  # SSbetween=14, SSwithin=6
    assert r.df == (2, 6)
    assert r.p == pytest.approx((6 / (6+2*7))**3)  # F(2,6) closed-form tail
    assert r.effect_size.value == pytest.approx(12/21)  # omega squared


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_paired_t_alignment_matches_hand_differences(layout):
    d = pd.DataFrame({"id": [40,10,50,20,30], "before": [10,30,20,15,40]})
    d["after"] = d.before + np.array([1,2,3,4,5])
    if layout == "wide":
        v = {"before": "before", "after": "after"}
    else:
        # Explicit ordered categories establish before -> after in the pivot.
        d = d.melt(id_vars="id", var_name="visit", value_name="y").sample(frac=1, random_state=8)
        d["visit"] = pd.Categorical(d.visit, categories=["before", "after"], ordered=True)
        v = {"outcome": "y", "subject": "id", "within": "visit"}
    r = run(Spec(test="paired_t", variables=v), d)
    assert r.n == 5
    assert r.estimates[0].value == pytest.approx(3)
    assert r.statistic == pytest.approx(3*sqrt(2))
    assert r.df == 4


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_linear_regression_orthogonal_residual_oracle(alternative):
    x = np.arange(-3., 4.)
    residual = np.array([1., -1., 0., 0., 0., -1., 1.])
    # Residual sum=0, x dot residual=0, SSE=4, Sxx=28, residual df=5.
    d = pd.DataFrame({"x": x, "y": 2 + .4*x + residual})
    statistic = .4 / sqrt((4/5)/28)
    r = run(Spec(test="simple_lin_reg", variables={"x": "x", "y": "y"}, alternative=alternative), d)
    assert r.estimates[0].value == pytest.approx(.4)
    assert r.estimates[1].value == pytest.approx(2)
    assert r.statistic == pytest.approx(statistic)
    assert r.p == pytest.approx(tail(statistic, 5, alternative))
    assert r.effect_size.value == pytest.approx((.4**2*28) / (.4**2*28+4))
    assert r.cross_check.agrees, "the secondary p-value must use the requested alternative too"


@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_partial_correlation_honors_direction_or_rejects_it(alternative):
    rng = np.random.default_rng(482)
    z, noise_x, noise_y = rng.normal(size=(3, 90))
    x, y = .8*z + noise_x, -.4*z + .5*noise_x + noise_y
    # Pearson scalar covariance formula; then the one-covariate identity.
    def corr(a, b):
        ac, bc = a-a.mean(), b-b.mean()
        return float(sum(ac*bc)/sqrt(sum(ac*ac)*sum(bc*bc)))
    rxy, rxz, ryz = corr(x,y), corr(x,z), corr(y,z)
    rpartial = (rxy-rxz*ryz)/sqrt((1-rxz**2)*(1-ryz**2))
    statistic = rpartial*sqrt(87/(1-rpartial**2))
    try:
        r = run(Spec(test="partial_corr", variables={"x": "x", "y": "y", "predictors": ["z"]},
                     alternative=alternative), pd.DataFrame({"x": x, "y": y, "z": z}))
    except SpecError:
        return  # Explicitly unsupported is acceptable; silently two-sided is not.
    assert r.effect_size.value == pytest.approx(rpartial)
    assert r.p == pytest.approx(tail(statistic, 87, alternative), rel=1e-7, abs=1e-12)


def multinomial_fixture():
    rows = []
    for age, counts in zip([-1., 0., 1.], [(15,10,8), (12,12,10), (9,14,12)]):
        for outcome, count in zip(["A", "B", "C"], counts):
            rows.extend([{"age": age, "outcome": outcome}] * count)
    return pd.DataFrame(rows)


def test_multinomial_formula_ignores_unused_columns_even_if_names_overlap():
    d = multinomial_fixture()
    spec = Spec(test="mnlogit", variables={"formula": "outcome ~ age"})
    a = run(spec, d)
    # 'a' is a substring of 'age', but it is not part of the formula design.
    b = run(spec, d.assign(a=np.nan))
    assert b.n == a.n == len(d)
    assert b.statistic == pytest.approx(a.statistic)
    assert [e.value for e in b.estimates] == pytest.approx([e.value for e in a.estimates])


def test_multinomial_removes_unused_categorical_levels_before_reporting():
    d = multinomial_fixture()
    plain = run(Spec(test="mnlogit", variables={"outcome": "outcome", "predictors": ["age"]}), d)
    d["outcome"] = pd.Categorical(d.outcome, categories=["unused", "A", "B", "C"])
    encoded = run(Spec(test="mnlogit", variables={"outcome": "outcome", "predictors": ["age"]}), d)
    assert encoded.extra["categories"] == ["A", "B", "C"]
    assert encoded.extra["reference"] == "A"
    assert [e.name for e in encoded.estimates] == [e.name for e in plain.estimates]
    assert [e.value for e in encoded.estimates] == pytest.approx([e.value for e in plain.estimates])


def test_kaplan_meier_product_limit_with_censoring_and_ties():
    # A: at t=1, two events among 6 => S=2/3; censor at 2; one event
    # among 3 at t=4 => S=4/9; remaining cases censored at t=5 and t=7.
    d = pd.DataFrame({"t": [1,1,2,4,5,7, 1,2,3,4,6,8],
                      "e": [1,1,0,1,0,0, 0,1,1,0,1,0], "g": ["A"]*6+["B"]*6})
    r = run(Spec(test="km_logrank", variables={"time": "t", "event": "e", "group": "g"},
                 params={"timepoints": [0,1,2,3,4,6]}), d)
    got = {e.name: e.value for e in r.estimates}
    for t, expected in [(0,1), (1,2/3), (2,2/3), (3,2/3), (4,4/9), (6,4/9)]:
        assert got[f"S({t})[A]"] == pytest.approx(expected)


def test_competing_incidence_from_hand_risk_sets():
    pytest.importorskip("lifelines")
    # At 1: S=5/6, CIF1=1/6. At 2: competing event, S=4/6.
    # At 3: censor. At 4: cause 1 among 3 => CIF1=1/6+(4/6)/3=7/18.
    d = pd.DataFrame({"t": [1,2,3,4,5,6], "event": [1,2,0,1,0,0]})
    r = run(Spec(test="competing", variables={"time": "t", "event": "event"},
                 decisions={"estimand": "cumulative_incidence", "event_of_interest": 1},
                 params={"timepoints": [0,1,3,4]}), d)
    got = {e.name: e.value for e in r.estimates}
    for t, expected in [(0,0), (1,1/6), (3,1/6), (4,7/18)]:
        assert got[f"cumulative incidence of 1 at {t}"] == pytest.approx(expected)
    assert r.n == {"subjects": 6, "events_of_interest": 2, "competing_events": 1, "censored": 3}


def test_logistic_group_model_matches_binomial_likelihood():
    # Group A: 10/40 successes; B: 30/50. The grouped model is saturated.
    d = pd.DataFrame({"y": [1]*10+[0]*30+[1]*30+[0]*20, "g": ["A"]*40+["B"]*50})
    r = run(Spec(test="logistic", variables={"outcome": "y", "predictors": ["g"]}), d)
    ll_full = 10*log(.25)+30*log(.75)+30*log(.6)+20*log(.4)
    ll_null = 40*log(40/90)+50*log(50/90)
    lr = 2*(ll_full-ll_null)
    estimates = {e.name: e.value for e in r.estimates}
    assert estimates["OR Intercept"] == pytest.approx(1/3)
    assert estimates["OR C(g)[T.B]"] == pytest.approx(4.5)
    assert r.statistic == pytest.approx(lr)
    assert r.p == pytest.approx(erfc(sqrt(lr/2)))  # chi-square(1) tail


def test_poisson_group_model_matches_closed_form_group_means():
    a, b = [0,1,1,2,1,1,2,0], [1,2,3,2,1,2,3,2]
    d = pd.DataFrame({"y": a+b, "g": ["A"]*8+["B"]*8})
    r = run(Spec(test="poisson_reg", variables={"outcome": "y", "predictors": ["g"]}), d)
    # Means 1 and 2, pooled mean 1.5; factorial terms cancel from LR.
    lr = 2*(8*log(1/1.5)+16*log(2/1.5))
    estimates = {e.name: e.value for e in r.estimates}
    assert estimates["RR Intercept"] == pytest.approx(1)
    assert estimates["RR C(g)[T.B]"] == pytest.approx(2)
    assert r.statistic == pytest.approx(lr)
    assert r.p == pytest.approx(erfc(sqrt(lr/2)))


@pytest.mark.parametrize("form", ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"])
def test_icc_from_direct_balanced_sums_of_squares(form):
    # Direct cell arithmetic supplies the oracle, without an ANOVA library.
    a = np.array([[2.,4.,3.], [5.,6.,5.], [7.,8.,9.], [3.,5.,4.]])
    n, k = a.shape
    row, col, grand = a.mean(axis=1), a.mean(axis=0), a.mean()
    msr = k*sum((row-grand)**2)/(n-1)
    msc = n*sum((col-grand)**2)/(k-1)
    mse = sum((a-row[:,None]-col[None,:]+grand).ravel()**2)/((n-1)*(k-1))
    msw = sum((a-row[:,None]).ravel()**2)/(n*(k-1))
    single = {"ICC1": (msr-msw)/(msr+(k-1)*msw),
              "ICC2": (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n),
              "ICC3": (msr-mse)/(msr+(k-1)*mse)}
    base = form.rstrip("k")
    expected = single[base]
    if form.endswith("k"):
        expected = k*expected/(1+(k-1)*expected)
    r = run(Spec(test="icc", variables={"columns": ["a","b","c"]},
                 decisions={"icc_form": form}), pd.DataFrame(a, columns=["a","b","c"]))
    assert r.effect_size.value == pytest.approx(expected)
    assert r.statistic == pytest.approx(msr/(msw if base == "ICC1" else mse))
    assert r.df == (n-1, n*(k-1) if base == "ICC1" else (n-1)*(k-1))
