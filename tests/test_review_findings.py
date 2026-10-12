"""Regression tests for the 13 findings of the third-party review of 2026-10-12 (codex_astra_review.md).

Each test pins the behaviour the review showed to be wrong, with expected values stated
independently of the code under test.
"""
import builtins

import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, SpecError, run
from whichtest.result import TestResult, compare

V = dict


# 1. label encoding must not depend on dtype, and missing labels are not negatives
@pytest.mark.parametrize("dtype", [int, float, bool])
def test_f1_binary_labels_are_dtype_invariant(dtype):
    a = np.array([0, 1, 0, 1], dtype=dtype)
    r = run(Spec(test="diag_binary", variables=V(outcome="y", x="p")), pd.DataFrame({"y": a, "p": a}))
    assert r.extra["table"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert r.estimates[0].value == 1.0  # sensitivity


def test_f1_missing_labels_are_excluded_jointly():
    d = pd.DataFrame({"y": [0, 1, None, 1, 0], "p": [0, 1, 1, None, 0]})
    r = run(Spec(test="diag_binary", variables=V(outcome="y", x="p")), d)
    assert r.extra["table"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 2}
    assert any("2 row(s)" in w for w in r.warnings)


def test_f1_string_labels_need_explicit_positive_for_both_columns():
    d = pd.DataFrame({"dx": ["pos", "neg", "pos", "neg"], "t": ["yes", "no", "no", "no"]})
    with pytest.raises(SpecError, match="positive_label"):
        run(Spec(test="diag_binary", variables=V(outcome="dx", x="t")), d)
    with pytest.raises(SpecError, match="test_positive_label"):
        run(Spec(test="diag_binary", variables=V(outcome="dx", x="t"), params={"positive_label": "pos"}), d)
    r = run(Spec(test="diag_binary", variables=V(outcome="dx", x="t"), params={"positive_label": "pos", "test_positive_label": "yes"}), d)
    assert r.extra["table"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}


def test_f1_roc_and_delong_share_the_typed_encoder():
    rng = np.random.default_rng(0)
    y = (rng.random(80) < 0.4)
    s = y + rng.normal(size=80)
    d_bool = pd.DataFrame({"y": y, "s": s, "s2": s + rng.normal(size=80)})
    d_int = d_bool.assign(y=y.astype(int))
    for t, extra in (("roc_auc", V(x="s")), ("delong", V(columns=["s", "s2"]))):
        a = run(Spec(test=t, variables=V(outcome="y", **extra), params={"n_boot": 50}), d_bool)
        b = run(Spec(test=t, variables=V(outcome="y", **extra), params={"n_boot": 50}), d_int)
        assert a.effect_size.value == pytest.approx(b.effect_size.value)


# 2. ordinal LR test must use the proper null and be invariant to centering
def test_f2_ordinal_lr_matches_statsmodels_and_is_centering_invariant():
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    x = pd.DataFrame({"x": np.random.default_rng(42).normal(3, 1, 300)})
    y = np.digitize(0.7 * x.x + np.random.default_rng(43).logistic(size=300), [1, 3])
    ref = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=0, maxiter=1000)
    s = Spec(test="ordinal_logit", variables=V(outcome="y", predictors=["x"]))
    r = run(s, x.assign(y=y))
    rc = run(s, x.assign(y=y, x=x.x - 3))
    assert r.statistic == pytest.approx(ref.llr, rel=1e-6)
    assert r.p == pytest.approx(ref.llr_pvalue, rel=1e-4)
    assert rc.statistic == pytest.approx(r.statistic, rel=1e-5)


# 3. missing counts or exposures leave the rate calculation entirely
def test_f3_missing_count_row_is_excluded_not_zeroed():
    d = pd.DataFrame({"y": [10, np.nan, 10, 10], "g": ["a", "a", "b", "b"], "ex": [1, 100, 1, 1]})
    r = run(Spec(test="poisson_2indep", variables=V(outcome="y", group="g", exposure="ex")), d)
    assert r.effect_size.value == pytest.approx(1.0)
    assert any("excluded" in w for w in r.warnings)
    d2 = pd.DataFrame({"y": [10, 5, 10, 10], "g": ["a", "a", "b", "b"], "ex": [1, np.nan, 1, 1]})
    r2 = run(Spec(test="poisson_2indep", variables=V(outcome="y", group="g", exposure="ex")), d2)
    assert r2.effect_size.value == pytest.approx(1.0)  # a: 10/1 after exclusion, b: 20/2; the naive nansum would give 1.5
    assert r2.n == {"a": 10, "b": 20}
    with pytest.raises(SpecError, match="positive"):
        run(Spec(test="poisson_2indep", variables=V(outcome="y", group="g", exposure="ex")), d2.assign(ex=[1, 0, 1, 1]))


# 4. Kaplan–Meier is a step function
def test_f4_km_timepoints_are_step_function_values():
    d = pd.DataFrame({"t": [1, 10, 20, 30, 2, 4, 6, 8], "e": [1] * 8, "g": ["a"] * 4 + ["b"] * 4})
    r = run(Spec(test="km_logrank", variables=V(time="t", event="e", group="g"), params={"timepoints": [0.5, 5, 10]}), d)
    got = {e.name: e.value for e in r.estimates if e.name.startswith("S(")}
    assert got["S(0.5)[a]"] == 1.0
    assert got["S(5)[a]"] == pytest.approx(0.75)
    assert got["S(10)[a]"] == pytest.approx(0.5)
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(d.t[d.g == "a"], d.e[d.g == "a"])
    assert got["S(5)[a]"] == pytest.approx(float(km.predict(5)))


# 5. non-finite values never count as agreement, and NaN inference is flagged
def test_f5_compare_rejects_non_finite_and_uses_strict_signs():
    assert not compare("m", "w", {"p": np.nan}, {"p": np.nan}, kind="approximate").agrees
    assert not compare("m", "w", {"x": np.inf}, {"x": np.inf}).agrees
    assert not compare("m", "w", {"direction": 0.0}, {"direction": -1.0}, kind="direction").agrees
    assert not compare("m", "w", {"direction": np.nan}, {"direction": -1.0}, kind="direction").agrees
    assert compare("m", "w", {"direction": 2.0}, {"direction": 0.5}, kind="direction").agrees
    r = TestResult(test="x", p=float("nan"), cross_check=compare("m", "w", {"p": 0.5}, {"p": 0.5}))
    assert r.significant is None and r.flagged and r.inference_failed
    assert r.to_dict()["p"] is None


# 6 & 7. ICC inference matches the chosen form; CI follows ci_level
@pytest.fixture
def icc_data():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"patient": np.repeat(range(30), 3), "rater": np.tile(["r1", "r2", "r3"], 30)})
    d["score"] = np.repeat(rng.normal(50, 10, 30), 3) + rng.normal(0, 3, 90) + np.tile([0, 5, 10], 30)
    return d


@pytest.mark.parametrize("form", ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"])
def test_f6_icc_inference_and_ci_match_pingouin(form, icc_data):
    import pingouin as pg

    r = run(Spec(test="icc", variables=V(outcome="score", subject="patient", within="rater"), decisions={"icc_form": form}), icc_data)
    assert r.cross_check.agrees, r.summary()
    ref = pg.intraclass_corr(data=icc_data, targets="patient", raters="rater", ratings="score")
    row = ref.iloc[["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"].index(form)]
    assert r.statistic == pytest.approx(float(row["F"]), rel=1e-6)
    assert r.df == (float(row["df1"]), float(row["df2"]))
    assert r.effect_size.ci[0] == pytest.approx(float(row["CI95"][0]), abs=0.0051)
    assert r.effect_size.ci[1] == pytest.approx(float(row["CI95"][1]), abs=0.0051)


def test_f7_icc_ci_follows_requested_level(icc_data):
    s = Spec(test="icc", variables=V(outcome="score", subject="patient", within="rater"), decisions={"icc_form": "ICC2"})
    r80 = run(s.model_copy(update={"ci_level": 0.8}), icc_data)
    r95 = run(s.model_copy(update={"ci_level": 0.95}), icc_data)
    assert r80.effect_size.ci_level == 0.8 and r95.effect_size.ci_level == 0.95
    assert r80.effect_size.ci[0] > r95.effect_size.ci[0] and r80.effect_size.ci[1] < r95.effect_size.ci[1]


def test_f7_aalen_johansen_ci_follows_requested_level(clinical):
    s = Spec(test="competing", variables=V(time="time", event="etype"), decisions={"estimand": "cumulative_incidence", "event_of_interest": 1}, params={"timepoints": [10]})
    r80 = run(s.model_copy(update={"ci_level": 0.8}), clinical)
    r95 = run(s.model_copy(update={"ci_level": 0.95}), clinical)
    e80 = next(e for e in r80.estimates if "at 10" in e.name)
    e95 = next(e for e in r95.estimates if "at 10" in e.name)
    assert e80.value == pytest.approx(e95.value)
    assert (e80.ci[1] - e80.ci[0]) < (e95.ci[1] - e95.ci[0])


# 8. every advertised working correlation runs, and row order does not matter
@pytest.mark.parametrize("wc", ["independence", "exchangeable", "ar1", "unstructured"])
def test_f8_every_working_correlation_runs(wc, longitudinal):
    s = Spec(test="gee_cont", variables=V(outcome="y", predictors=["visit", "group"], subject="id", time="visit"), decisions={"working_correlation": wc})
    a = run(s, longitudinal)
    b = run(s, longitudinal.sample(frac=1, random_state=3))
    assert a.cross_check.agrees
    assert [e.value for e in a.estimates] == pytest.approx([e.value for e in b.estimates], rel=1e-6)


def test_f8_unstructured_requires_time(longitudinal):
    with pytest.raises(SpecError, match="variables.time"):
        run(Spec(test="gee_cont", variables=V(outcome="y", predictors=["visit"], subject="id"), decisions={"working_correlation": "unstructured"}), longitudinal)


# 9. competing-risk event codes are compared as numbers
def test_f9_float_event_codes(clinical):
    s = Spec(test="competing", variables=V(time="time", event="etype", predictors=["age", "group"]), decisions={"estimand": "cause_specific_hazard", "event_of_interest": 1})
    r_int = run(s, clinical)
    r_float = run(s, clinical.assign(etype=clinical.etype.astype(float)))
    assert r_float.n == r_int.n
    assert r_int.n["censored"] == int((clinical.etype == 0).sum())
    assert r_float.statistic == pytest.approx(r_int.statistic)
    with pytest.raises(SpecError, match="censoring code"):
        run(s.model_copy(update={"decisions": {"estimand": "cumulative_incidence", "event_of_interest": 0}}), clinical)


# 10. missing multinomial outcomes are dropped, not modelled as a category
def test_f10_mnlogit_missing_outcome(clinical):
    d = clinical.copy()
    d.loc[d.index[:2], "ynom"] = None
    r = run(Spec(test="mnlogit", variables=V(outcome="ynom", predictors=["age", "group"])), d)
    assert r.n == len(clinical) - 2
    assert r.extra["categories"] == ["A", "B", "C"]


# 11. one-sided intervals and NaNs format
def test_f11_one_sided_summary_formats():
    r = run(Spec(test="one_sample_t", variables=V(outcome="y"), alternative="greater"), pd.DataFrame({"y": [1, 2, 3, 4, 5]}))
    assert "inf]" in r.summary()
    r2 = run(Spec(test="one_sample_t", variables=V(outcome="y"), alternative="less"), pd.DataFrame({"y": [1, 2, 3, 4, 5]}))
    assert "[95% CI -inf" in r2.summary()


# 12. a cross-check that cannot run is an explicit failed cross-check
def test_f12_missing_cross_check_dependency_flags_the_result(monkeypatch):
    real = builtins.__import__

    def blocker(name, *a, **k):
        if name.startswith("pingouin"):
            raise ImportError("pingouin")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocker)
    r = run(Spec(test="friedman", variables=V(columns=["a", "b", "c"])), pd.DataFrame(np.random.default_rng(0).normal(size=(15, 3)), columns=list("abc")))
    assert r.cross_check is not None and not r.cross_check.agrees
    assert "NOT RUN" in r.cross_check.detail
    assert r.flagged and any("cross-check not run" in w for w in r.warnings)


# 13. validation branches raise SpecError, not NameError
def test_f13_validation_errors_are_spec_errors():
    with pytest.raises(SpecError, match="paired_t"):
        run(Spec(test="rm_anova", variables=V(columns=["a", "b"])), pd.DataFrame({"a": [1.0, 2, 3], "b": [2.0, 3, 4]}))
    with pytest.raises(SpecError, match="two factors"):
        run(Spec(test="factorial_anova", variables=V(outcome="y", predictors=["a"]), decisions={"ss_type": 2, "include_interaction": False}), pd.DataFrame({"y": [1.0, 2, 3, 4], "a": list("pqpq")}))
