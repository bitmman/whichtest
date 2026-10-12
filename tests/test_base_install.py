"""A base install (no ``cross`` extra) must still run every procedure and must say when a
cross-check could not run.

In a full dev environment this test blocks the optional imports so the same contract is
exercised; in CI's base-only job the libraries are genuinely absent.
"""
from __future__ import annotations

import builtins
import importlib.util

import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, run
from whichtest.taxonomy import load_taxonomy

OPTIONAL = ("pingouin", "lifelines", "krippendorff", "scikit_posthocs", "semopy")


@pytest.fixture(autouse=True)
def _no_optional_libraries(monkeypatch):
    present = [m for m in OPTIONAL if importlib.util.find_spec(m) is not None]
    if not present:
        yield  # genuinely a base install
        return
    real = builtins.__import__

    def blocker(name, *a, **k):
        if name.split(".")[0] in OPTIONAL:
            raise ImportError(f"{name} blocked to simulate a base install")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocker)
    yield


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 120
    d = pd.DataFrame({"age": rng.normal(50, 10, n), "group": rng.choice(["a", "b"], n), "arm": rng.choice(["p", "q", "r"], n), "x": rng.normal(size=n)})
    lin = 0.03 * (d.age - 50) + 0.8 * (d.group == "b") + 0.5 * d.x
    d["y"] = lin + rng.normal(size=n)
    d["ybin"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    d["ynom"] = rng.choice(["A", "B", "C"], n)
    d["ycount"] = rng.poisson(np.exp(0.4 + 0.3 * (d.group == "b")))
    d["exposure"] = rng.uniform(0.5, 2, n)
    d["before"] = rng.normal(10, 2, n)
    d["after"] = d.before + rng.normal(0.5, 1, n)
    d["t1"], d["t2"], d["t3"] = d.before + rng.normal(0, 1, n), d.before + rng.normal(0.4, 1, n), d.before + rng.normal(0.8, 1, n)
    d["b1"] = rng.integers(0, 2, n)
    d["b2"] = np.where(rng.random(n) < 0.7, d.b1, 1 - d.b1)
    d["b3"] = np.where(rng.random(n) < 0.6, d.b1, 1 - d.b1)
    d["r1"] = rng.integers(0, 3, n)
    d["r2"] = np.where(rng.random(n) < 0.7, d.r1, rng.integers(0, 3, n))
    d["r3"] = np.where(rng.random(n) < 0.6, d.r1, rng.integers(0, 3, n))
    d["time"] = rng.exponential(np.where(d.group == "b", 15, 10))
    d["event"] = (rng.random(n) > 0.3).astype(int)
    d["etype"] = np.where(d.event == 1, rng.choice([1, 2], n), 0)
    d["site"] = rng.integers(0, 40, n)
    d["m2"] = d.before + 1 + rng.normal(0, 2, n)
    d["score"] = d.ybin + rng.normal(size=n)
    d["score2"] = 0.5 * d.ybin + rng.normal(size=n)
    d["pred"] = (d.score > 0.5).astype(int)
    d["y2"] = 0.5 * d.y + rng.normal(size=n)
    d["y3"] = rng.normal(size=n)
    d["zc"] = np.where(rng.random(n) < 0.3, 0, d.ycount)
    f1 = rng.normal(size=n)
    for i in range(4):
        d[f"i{i}"] = f1 * (1 if i < 2 else 0.2) + rng.normal(size=n)
    return d


@pytest.fixture(scope="module")
def long():
    rng = np.random.default_rng(4)
    ns, nt = 40, 3
    d = pd.DataFrame({"id": np.repeat(range(ns), nt), "visit": np.tile(range(nt), ns)})
    d["group"] = np.repeat(rng.choice(["a", "b"], ns), nt)
    d["y"] = 0.4 * d.visit + np.repeat(rng.normal(size=ns), nt) + rng.normal(0, 0.7, len(d))
    d["ybin"] = (rng.random(len(d)) < 1 / (1 + np.exp(-(d.y - 0.5)))).astype(int)
    d["cnt"] = rng.poisson(np.exp(0.2 + 0.2 * d.visit))
    d["yord"] = pd.cut(d.y + rng.logistic(size=len(d)), [-np.inf, 0, 1, np.inf], labels=[0, 1, 2]).astype(int)
    d["ynom"] = rng.choice(["x", "y", "z"], len(d))
    return d


V = dict
SPECS = {
    "one_sample_t": Spec(test="one_sample_t", variables=V(outcome="before")),
    "indep_t": Spec(test="indep_t", variables=V(outcome="y", group="group")),
    "oneway_anova": Spec(test="oneway_anova", variables=V(outcome="y", group="arm")),
    "simple_lin_reg": Spec(test="simple_lin_reg", variables=V(x="age", y="y")),
    "multiple_reg": Spec(test="multiple_reg", variables=V(outcome="y", predictors=["group", "age"])),
    "paired_t": Spec(test="paired_t", variables=V(before="before", after="after")),
    "wilcoxon_signed": Spec(test="wilcoxon_signed", variables=V(before="before", after="after")),
    "friedman": Spec(test="friedman", variables=V(columns=["t1", "t2", "t3"])),
    "one_sample_wilcoxon": Spec(test="one_sample_wilcoxon", variables=V(outcome="before")),
    "mann_whitney": Spec(test="mann_whitney", variables=V(outcome="y", group="group")),
    "kruskal": Spec(test="kruskal", variables=V(outcome="y", group="arm")),
    "spearman": Spec(test="spearman", variables=V(x="age", y="y")),
    "mcnemar": Spec(test="mcnemar", variables=V(before="b1", after="b2")),
    "cochran_q": Spec(test="cochran_q", variables=V(columns=["b1", "b2", "b3"])),
    "binom_test": Spec(test="binom_test", variables=V(outcome="ybin")),
    "two_by_two": Spec(test="two_by_two", variables=V(row="group", col="ybin")),
    "chi_sq_indep": Spec(test="chi_sq_indep", variables=V(row="arm", col="ynom")),
    "logistic": Spec(test="logistic", variables=V(outcome="ybin", predictors=["age", "group"])),
    "stuart_maxwell": Spec(test="stuart_maxwell", variables=V(before="r1", after="r2")),
    "chi_gof": Spec(test="chi_gof", variables=V(outcome="ynom")),
    "mnlogit": Spec(test="mnlogit", variables=V(outcome="ynom", predictors=["age"])),
    "poisson_2indep": Spec(test="poisson_2indep", variables=V(outcome="ycount", group="group", exposure="exposure")),
    "poisson_reg": Spec(test="poisson_reg", variables=V(outcome="ycount", predictors=["age", "group"])),
    "km_logrank": Spec(test="km_logrank", variables=V(time="time", event="event", group="group")),
    "pearson": Spec(test="pearson", variables=V(x="age", y="y")),
    "point_biserial": Spec(test="point_biserial", variables=V(outcome="y", group="group")),
    "cohen_kappa": Spec(test="cohen_kappa", variables=V(columns=["r1", "r2"])),
    "fleiss_kappa": Spec(test="fleiss_kappa", variables=V(columns=["r1", "r2", "r3"]), params={"n_boot": 50}),
    "rm_anova": Spec(test="rm_anova", variables=V(columns=["t1", "t2", "t3"])),
    "ordinal_logit": Spec(test="ordinal_logit", variables=V(outcome="r1", predictors=["x"])),
    "cox": Spec(test="cox", variables=V(time="time", event="event", predictors=["age", "group"])),
    "manova": Spec(test="manova", variables=V(columns=["y", "y2"], group="group")),
    "mancova": Spec(test="mancova", variables=V(columns=["y", "y2"], group="group", predictors=["age"])),
    "mv_reg": Spec(test="mv_reg", variables=V(columns=["y", "y3"], predictors=["x", "age"])),
    "partial_corr": Spec(test="partial_corr", variables=V(x="x", y="y", predictors=["age"])),
    "bland_altman": Spec(test="bland_altman", variables=V(columns=["before", "m2"])),
    "diag_binary": Spec(test="diag_binary", variables=V(outcome="ybin", x="pred")),
    "roc_auc": Spec(test="roc_auc", variables=V(outcome="ybin", x="score")),
    "delong": Spec(test="delong", variables=V(outcome="ybin", columns=["score", "score2"]), params={"n_boot": 50}),
    "factorial_anova": Spec(test="factorial_anova", variables=V(outcome="y", predictors=["group", "arm"]), decisions={"ss_type": 2, "include_interaction": True}),
    "zip": Spec(test="zip", variables=V(outcome="zc", predictors=["group"]), decisions={"count_distribution": "poisson", "inflation_formula": "1"}),
    "competing": Spec(test="competing", variables=V(time="time", event="etype", predictors=["age"]), decisions={"estimand": "cause_specific_hazard", "event_of_interest": 1}),
    "cox_cluster": Spec(test="cox_cluster", variables=V(time="time", event="event", predictors=["age"], cluster="site"), decisions={"approach": "cluster_robust"}),
    "factor_sem": Spec(test="factor_sem", variables=V(columns=["i0", "i1", "i2", "i3"]), decisions={"approach": "exploratory", "n_factors": 1, "rotation": "none"}),
    "icc": Spec(test="icc", variables=V(columns=["t1", "t2", "t3"]), decisions={"icc_form": "ICC2"}),
}
LONG_SPECS = {
    "lmm": Spec(test="lmm", variables=V(outcome="y", predictors=["visit", "group"], subject="id"), decisions={"re_formula": "1", "reml": True}),
    "gee_cont": Spec(test="gee_cont", variables=V(outcome="y", predictors=["visit"], subject="id"), decisions={"working_correlation": "exchangeable"}),
    "gee_logit": Spec(test="gee_logit", variables=V(outcome="ybin", predictors=["visit"], subject="id"), decisions={"working_correlation": "independence"}),
    "gee_poisson": Spec(test="gee_poisson", variables=V(outcome="cnt", predictors=["visit"], subject="id"), decisions={"working_correlation": "exchangeable"}),
    "ordinal_gee": Spec(test="ordinal_gee", variables=V(outcome="yord", predictors=["visit"], subject="id"), decisions={"working_correlation": "independence"}),
    "nominal_gee": Spec(test="nominal_gee", variables=V(outcome="ynom", predictors=["visit"], subject="id"), decisions={"working_correlation": "independence"}),
}
# Cross-checks that live in an optional library: these must report NOT RUN in a base install.
OPTIONAL_CROSS = {"friedman", "cochran_q", "fleiss_kappa", "km_logrank", "rm_anova", "cox", "cox_cluster", "partial_corr", "icc", "competing"}


def test_every_taxonomy_test_has_a_spec_here():
    assert set(load_taxonomy().tests) == set(SPECS) | set(LONG_SPECS)


@pytest.mark.parametrize("test_id", sorted(SPECS) + sorted(LONG_SPECS))
def test_runs_without_optional_libraries(test_id, data, long):
    spec = SPECS.get(test_id) or LONG_SPECS[test_id]
    r = run(spec, long if test_id in LONG_SPECS else data)
    assert r.cross_check is not None, "a result must always carry a cross-check object"
    if test_id in OPTIONAL_CROSS:
        assert not r.cross_check.agrees and "NOT RUN" in r.cross_check.detail, r.cross_check
        assert r.flagged and any("cross-check not run" in w for w in r.warnings)
    else:
        assert r.cross_check.agrees, r.summary()
    r.to_json()
    r.summary()


def test_cumulative_incidence_needs_lifelines(data):
    """The primary estimator (Aalen–Johansen) lives in lifelines: without it the estimand is refused, not approximated."""
    from whichtest import SpecError

    spec = Spec(test="competing", variables=V(time="time", event="etype"), decisions={"estimand": "cumulative_incidence", "event_of_interest": 1})
    if importlib.util.find_spec("lifelines") is None or True:  # the autouse fixture blocks the import either way
        with pytest.raises(SpecError, match="lifelines"):
            run(spec, data)
