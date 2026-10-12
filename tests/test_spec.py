import pytest

from whichtest import Design, Spec, SpecError, Variables, run


def test_from_design_fills_taxonomy_fields():
    s = Spec.from_design(
        {"goal": "single_outcome", "outcome_type": "continuous", "structure": "independent", "predictors": "one_factor_2_levels"},
        variables={"outcome": "y", "group": "g"},
    )
    assert s.test == "indep_t"
    assert s.status == "ready"
    assert s.assumption_checks == ["normality_by_group", "independence", "variance_info"]
    assert s.fallback == "mann_whitney"
    assert "statsmodels" in s.cross_check


def test_informational_fields_must_agree():
    with pytest.raises(SpecError):
        Spec(test="indep_t", status="helper").complete()


def test_fallback_on_chain_is_allowed():
    d = Design(goal="single_outcome", outcome_type="continuous", structure="independent", predictors="one_factor_2_levels")
    s = Spec(test="mann_whitney", design=d).complete()
    assert s.resolved_test == "indep_t"
    with pytest.raises(SpecError):
        Spec(test="pearson", design=d).complete()


def test_unknown_test():
    with pytest.raises(SpecError):
        Spec(test="nope").complete()


def test_missing_bindings_give_clear_error(clinical):
    with pytest.raises(SpecError, match="needs variables"):
        run(Spec(test="indep_t"), clinical)


def test_judgment_tests_refuse_without_decisions(clinical):
    with pytest.raises((SpecError, NotImplementedError)):
        run(Spec(test="lmm", variables=Variables(outcome="y")), clinical)


def test_roundtrip_json():
    s = Spec.from_design({"goal": "association", "variable_types": "both_continuous_linear"}, variables={"x": "a", "y": "b"})
    s2 = Spec.model_validate_json(s.to_json())
    assert s2 == s
