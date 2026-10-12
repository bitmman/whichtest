"""Every ready test runs on synthetic data and its cross-check agrees."""
import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, available, run
from whichtest.taxonomy import load_taxonomy

V = dict

SPECS = {
    "one_sample_t": Spec(test="one_sample_t", variables=V(outcome="before"), params={"mu0": 9.5}),
    "indep_t": Spec(test="indep_t", variables=V(outcome="y", group="group")),
    "oneway_anova": Spec(test="oneway_anova", variables=V(outcome="y", group="arm")),
    "simple_lin_reg": Spec(test="simple_lin_reg", variables=V(x="age", y="y")),
    "multiple_reg": Spec(test="multiple_reg", variables=V(outcome="y", predictors=["group", "age", "x"])),
    "paired_t": Spec(test="paired_t", variables=V(before="before", after="after")),
    "wilcoxon_signed": Spec(test="wilcoxon_signed", variables=V(before="before", after="after")),
    "friedman": Spec(test="friedman", variables=V(columns=["t1", "t2", "t3"])),
    "one_sample_wilcoxon": Spec(test="one_sample_wilcoxon", variables=V(outcome="before"), params={"m0": 9.5}),
    "mann_whitney": Spec(test="mann_whitney", variables=V(outcome="y_skew", group="group")),
    "kruskal": Spec(test="kruskal", variables=V(outcome="y_skew", group="arm")),
    "spearman": Spec(test="spearman", variables=V(x="age", y="y")),
    "mcnemar": Spec(test="mcnemar", variables=V(before="b1", after="b2")),
    "cochran_q": Spec(test="cochran_q", variables=V(columns=["b1", "b2", "b3"])),
    "binom_test": Spec(test="binom_test", variables=V(outcome="ybin"), params={"p0": 0.5}),
    "two_by_two": Spec(test="two_by_two", variables=V(row="group", col="ybin")),
    "chi_sq_indep": Spec(test="chi_sq_indep", variables=V(row="arm", col="ynom")),
    "logistic": Spec(test="logistic", variables=V(outcome="ybin", predictors=["age", "group", "x"])),
    "stuart_maxwell": Spec(test="stuart_maxwell", variables=V(before="o1", after="o2")),
    "chi_gof": Spec(test="chi_gof", variables=V(outcome="ynom"), params={"expected": [0.5, 0.3, 0.2]}),
    "mnlogit": Spec(test="mnlogit", variables=V(outcome="ynom", predictors=["age", "group"])),
    "poisson_2indep": Spec(test="poisson_2indep", variables=V(outcome="ycount", group="group", exposure="exposure")),
    "poisson_reg": Spec(test="poisson_reg", variables=V(outcome="ycount", predictors=["age", "group"], exposure="exposure")),
    "km_logrank": Spec(test="km_logrank", variables=V(time="time", event="event", group="group")),
    "pearson": Spec(test="pearson", variables=V(x="age", y="y")),
    "point_biserial": Spec(test="point_biserial", variables=V(outcome="y", group="group")),
    "cohen_kappa": Spec(test="cohen_kappa", variables=V(columns=["r1", "r2"])),
    "fleiss_kappa": Spec(test="fleiss_kappa", variables=V(columns=["r1", "r2", "r3"]), params={"n_boot": 200}),
}

HELPER_SPECS = {
    "rm_anova": Spec(test="rm_anova", variables=V(columns=["t1", "t2", "t3"])),
    "ordinal_logit": Spec(test="ordinal_logit", variables=V(outcome="yord", predictors=["x", "group"])),
    "cox": Spec(test="cox", variables=V(time="time", event="event", predictors=["age", "group"])),
    "manova": Spec(test="manova", variables=V(columns=["y1", "y2", "y3"], group="group")),
    "mancova": Spec(test="mancova", variables=V(columns=["y1", "y2"], group="group", predictors=["age"])),
    "mv_reg": Spec(test="mv_reg", variables=V(columns=["y1", "y3"], predictors=["x", "age"])),
    "partial_corr": Spec(test="partial_corr", variables=V(x="x", y="y1", predictors=["age", "bmi"])),
    "bland_altman": Spec(test="bland_altman", variables=V(columns=["m1", "m2"])),
    "diag_binary": Spec(test="diag_binary", variables=V(outcome="truth", x="test_pos")),
    "roc_auc": Spec(test="roc_auc", variables=V(outcome="truth", x="score_a"), params={"positive_label": 1, "threshold": 0.5}),
    "delong": Spec(test="delong", variables=V(outcome="truth", columns=["score_a", "score_b"]), params={"n_boot": 300}),
}
ALL_SPECS = {**SPECS, **HELPER_SPECS}


def test_all_ready_and_helper_tests_are_implemented_and_covered():
    tax = load_taxonomy()
    ready, helper = set(tax.by_status("ready")), set(tax.by_status("helper"))
    assert ready | helper <= set(available())
    assert ready == set(SPECS)
    assert helper == set(HELPER_SPECS)


@pytest.mark.parametrize("test_id", sorted(ALL_SPECS))
def test_runs_and_cross_check_agrees(test_id, clinical):
    r = run(ALL_SPECS[test_id], clinical)
    assert r.test == test_id
    assert r.status in ("ready", "helper")
    assert r.cross_check is not None, "every test must have a cross-check"
    assert r.cross_check.agrees, r.summary()
    if r.p is not None:
        assert 0 <= r.p <= 1
    assert r.checks, "every test reports its assumption checks"
    d = r.to_dict()
    assert d["test"] == test_id
    r.to_json()  # serialisable
    assert r.summary()


# --- exact identities from CLAUDE.md: good first unit tests ---------------------


def test_paired_t_equals_one_sample_on_differences(clinical):
    from scipy import stats

    r = run(SPECS["paired_t"], clinical)
    d = clinical.after - clinical.before
    ref = stats.ttest_1samp(d, 0)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-12)


def test_spearman_equals_pearson_on_ranks(clinical):
    from scipy import stats

    r = run(SPECS["spearman"], clinical)
    ref = stats.pearsonr(stats.rankdata(clinical.age), stats.rankdata(clinical.y))
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)


def test_mann_whitney_u_is_auc(clinical):
    from sklearn.metrics import roc_auc_score

    r = run(SPECS["mann_whitney"], clinical)
    g = clinical.group
    first = list(r.n)[0]  # U is computed for the first group in order of appearance
    n1, n2 = (g == first).sum(), (g != first).sum()
    auc = roc_auc_score((g == first).astype(int), clinical.y_skew)
    assert r.statistic / (n1 * n2) == pytest.approx(auc, rel=1e-12)


def test_exact_mcnemar_is_binomial_on_discordant():
    from scipy import stats

    r = run(Spec(test="mcnemar", params={"table": [[20, 5], [12, 30]], "exact": True}))
    assert r.p == pytest.approx(stats.binomtest(5, 17, 0.5).pvalue, rel=1e-12)


def test_logit_equals_binomial_glm(clinical):
    r = run(SPECS["logistic"], clinical)
    for c in r.cross_check.comparisons:
        assert c.primary == pytest.approx(c.secondary, rel=1e-6, abs=1e-6)


def test_point_biserial_is_students_t(clinical):
    from scipy import stats

    r = run(SPECS["point_biserial"], clinical)
    a, b = clinical.y[clinical.group == "trt"], clinical.y[clinical.group == "ctl"]
    assert r.p == pytest.approx(stats.ttest_ind(a, b).pvalue, rel=1e-10)


# --- behaviour ---------------------------------------------------------------


def test_fallback_suggested_when_checks_fail(clinical):
    r = run(Spec(test="indep_t", variables=V(outcome="y_skew", group="group")), clinical)
    assert any(c.key == "normality_by_group" and c.failed for c in r.checks)
    assert r.fallback["to"] == "mann_whitney"
    assert r.flagged


def test_two_by_two_switches_to_fisher_on_sparse_table():
    r = run(Spec(test="two_by_two", params={"table": [[2, 7], [8, 1]]}))
    assert r.statistic_name.startswith("Fisher")
    assert "fisher" in r.extra


def test_poisson_reg_switches_to_negative_binomial(clinical):
    r = run(Spec(test="poisson_reg", variables=V(outcome="ynb", predictors=["group"])), clinical)
    assert r.name.startswith("negative binomial")
    assert any(c.key == "overdispersion" and c.failed for c in r.checks)


def test_long_format_repeated_measures(clinical):
    long = clinical[["t1", "t2", "t3"]].reset_index().melt(id_vars="index", var_name="time", value_name="val")
    wide = run(SPECS["friedman"], clinical)
    lng = run(Spec(test="friedman", variables=V(outcome="val", subject="index", within="time")), long)
    assert lng.statistic == pytest.approx(wide.statistic)


def test_known_answer_welch():
    """Textbook check: two tiny groups with a hand-computable Welch t."""
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "g": ["a"] * 5 + ["b"] * 5})
    r = run(Spec(test="indep_t", variables=V(outcome="y", group="g")), df)
    # means 3 and 8, var 2.5 each → se = sqrt(1) = 1, t = -5, df = 8
    assert r.statistic == pytest.approx(-5.0)
    assert r.df == pytest.approx(8.0)


# --- helper-test identities ------------------------------------------------


def test_roc_auc_equals_u_over_n1n0(clinical):
    from scipy import stats

    r = run(HELPER_SPECS["roc_auc"], clinical)
    y, s = clinical.truth.to_numpy(), clinical.score_a.to_numpy()
    u = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert r.effect_size.value == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()), rel=1e-12)


def test_cox_matches_lifelines_to_four_decimals(clinical):
    r = run(HELPER_SPECS["cox"], clinical)
    for c in r.cross_check.comparisons:
        assert c.primary == pytest.approx(c.secondary, rel=1e-4, abs=1e-6)


def test_partial_corr_equals_residual_correlation(clinical):
    import statsmodels.api as sm
    from scipy import stats

    r = run(HELPER_SPECS["partial_corr"], clinical)
    Z = sm.add_constant(clinical[["age", "bmi"]])
    rx = sm.OLS(clinical.x, Z).fit().resid
    ry = sm.OLS(clinical.y1, Z).fit().resid
    assert r.effect_size.value == pytest.approx(stats.pearsonr(rx, ry).statistic, rel=1e-10)


def test_rm_anova_flags_sphericity_and_reports_gg(clinical):
    r = run(HELPER_SPECS["rm_anova"], clinical)
    assert "greenhouse_geisser" in r.extra
    assert r.extra["greenhouse_geisser"]["p"] is not None


def test_ordinal_logit_with_category_labels(clinical):
    d = clinical.assign(sev=clinical.yord.map({0: "mild", 1: "moderate", 2: "severe"}))
    r = run(Spec(test="ordinal_logit", variables=V(outcome="sev", predictors=["x", "group"]), params={"levels": ["mild", "moderate", "severe"]}), d)
    ref = run(HELPER_SPECS["ordinal_logit"], clinical)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-6)


def test_diag_binary_requires_explicit_positive_label_for_strings(clinical):
    d = clinical.assign(dx=clinical.truth.map({0: "healthy", 1: "disease"}))
    with pytest.raises(Exception, match="positive_label"):
        run(Spec(test="diag_binary", variables=V(outcome="dx", x="test_pos")), d)
    r = run(Spec(test="diag_binary", variables=V(outcome="dx", x="test_pos"), params={"positive_label": "disease"}), d)
    ref = run(HELPER_SPECS["diag_binary"], clinical)
    assert r.estimates[0].value == pytest.approx(ref.estimates[0].value)
