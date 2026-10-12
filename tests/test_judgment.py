"""Judgment tests: gated on spec.decisions, and their cross-checks agree once the decisions are given."""
import pytest

from whichtest import Spec, SpecError, available, run
from whichtest.registry import PROCEDURES, _load
from whichtest.taxonomy import load_taxonomy

V = dict

LONG = {
    "lmm": Spec(test="lmm", variables=V(outcome="y", predictors=["visit", "group"], subject="id"), decisions={"re_formula": "1", "reml": True}),
    "gee_cont": Spec(test="gee_cont", variables=V(outcome="y", predictors=["visit", "group"], subject="id"), decisions={"working_correlation": "exchangeable"}),
    "gee_logit": Spec(test="gee_logit", variables=V(outcome="ybin", predictors=["visit", "group"], subject="id"), decisions={"working_correlation": "ar1"}),
    "gee_poisson": Spec(test="gee_poisson", variables=V(outcome="cnt", predictors=["visit", "group"], subject="id", exposure="months"), decisions={"working_correlation": "exchangeable"}),
    "ordinal_gee": Spec(test="ordinal_gee", variables=V(outcome="yord", predictors=["visit", "group"], subject="id"), decisions={"working_correlation": "global_odds_ratio"}),
    "nominal_gee": Spec(test="nominal_gee", variables=V(outcome="ynom", predictors=["visit", "group"], subject="id"), decisions={"working_correlation": "independence"}),
}
CLIN = {
    "factorial_anova": Spec(test="factorial_anova", variables=V(outcome="y", predictors=["group", "factor_b"]), decisions={"ss_type": 3, "include_interaction": True}),
    "zip": Spec(test="zip", variables=V(outcome="zcnt", predictors=["group"]), decisions={"count_distribution": "negbin", "inflation_formula": "1"}),
    "competing": Spec(test="competing", variables=V(time="time", event="etype", predictors=["age", "group"]), decisions={"estimand": "cause_specific_hazard", "event_of_interest": 1}, params={"timepoints": [5, 10]}),
    "cox_cluster": Spec(test="cox_cluster", variables=V(time="time", event="event", predictors=["age", "group"], cluster="site"), decisions={"approach": "cluster_robust"}),
    "factor_sem": Spec(test="factor_sem", variables=V(columns=[f"item{i}" for i in range(6)]), decisions={"approach": "exploratory", "n_factors": 2, "rotation": "varimax"}),
}
RATE = {
    "icc": Spec(test="icc", variables=V(outcome="score", subject="patient", within="rater"), decisions={"icc_form": "ICC(2,1)"}),
}
ALL = {**LONG, **CLIN, **RATE}


def test_every_taxonomy_test_is_implemented():
    tax = load_taxonomy()
    assert set(tax.tests) == set(available())
    assert set(tax.by_status("judgment")) == set(ALL)


def test_every_judgment_test_declares_decisions():
    _load()
    for tid in load_taxonomy().by_status("judgment"):
        assert PROCEDURES[tid].decisions, f"{tid} must declare the decisions a human has to make"


@pytest.mark.parametrize("test_id", sorted(ALL))
def test_refuses_without_decisions(test_id, clinical, longitudinal, ratings):
    spec = ALL[test_id].model_copy(update={"decisions": {}})
    data = longitudinal if test_id in LONG else ratings if test_id in RATE else clinical
    with pytest.raises(SpecError, match="judgment test"):
        run(spec, data)


@pytest.mark.parametrize("test_id", sorted(ALL))
def test_runs_with_decisions_and_cross_check_agrees(test_id, clinical, longitudinal, ratings):
    data = longitudinal if test_id in LONG else ratings if test_id in RATE else clinical
    r = run(ALL[test_id], data)
    assert r.status == "judgment"
    assert r.cross_check is not None and r.cross_check.agrees, r.summary()
    assert r.extra["decisions"] == ALL[test_id].decisions, "the human decisions must be echoed in the result"
    assert "decisions (set by a human)" in r.summary()
    r.to_json()


def test_icc_manual_formula_matches_pingouin_for_every_form(ratings):
    for form in ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]:
        r = run(RATE["icc"].model_copy(update={"decisions": {"icc_form": form}}), ratings)
        assert r.cross_check.agrees, form


def test_cox_cluster_stratified(clinical):
    r = run(Spec(test="cox_cluster", variables=V(time="time", event="event", predictors=["age", "group"], strata="stage"), decisions={"approach": "stratify"}), clinical)
    assert r.cross_check.agrees


def test_competing_cumulative_incidence_is_below_naive(clinical):
    r = run(CLIN["competing"].model_copy(update={"decisions": {"estimand": "cumulative_incidence", "event_of_interest": 1}}), clinical)
    aj = next(e for e in r.estimates if "end of follow-up" in e.name and "Aalen" in (e.note or "")).value
    naive = next(e for e in r.estimates if "naive" in e.name).value
    assert aj <= naive


def test_factorial_anova_shows_both_ss_types(clinical):
    r = run(CLIN["factorial_anova"], clinical)
    assert "type2" in r.extra and "type3" in r.extra


def test_zip_poisson_variant(clinical):
    r = run(CLIN["zip"].model_copy(update={"decisions": {"count_distribution": "poisson", "inflation_formula": "group"}}), clinical)
    assert r.name.startswith("zero-inflated Poisson")
