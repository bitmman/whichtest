"""Regression tests for the round-three review (R3-1 … R3-8) beyond the reviewer's own suite."""
import json

import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, SpecError, run
from whichtest.cli import main
from whichtest.registry import PROCEDURES, _load, preflight

V = dict


def test_every_judgment_decision_is_typed_and_documented():
    _load()
    from whichtest.taxonomy import load_taxonomy

    for tid in load_taxonomy().by_status("judgment"):
        for d in PROCEDURES[tid].decisions:
            assert d.doc and d.kind in ("choice", "bool", "int", "str", "code"), (tid, d.key)
            if d.kind == "choice":
                assert d.choices, (tid, d.key)


def test_unknown_decision_key_is_rejected(longitudinal):
    s = Spec(test="lmm", variables=V(outcome="y", predictors=["visit"], subject="id"), decisions={"re_formula": "1", "reml": True, "remel": False})
    with pytest.raises(SpecError, match="does not take decisions \\['remel'\\]"):
        run(s, longitudinal)


def test_decisions_for_a_ready_test_are_rejected(clinical):
    with pytest.raises(SpecError, match="does not take decisions"):
        run(Spec(test="indep_t", variables=V(outcome="y", group="group"), decisions={"reml": True}), clinical)


@pytest.mark.parametrize("value", ["true", 1, 1.0, "yes"])
def test_bool_decisions_need_json_booleans(value, clinical):
    s = Spec(test="factorial_anova", variables=V(outcome="y", predictors=["group", "factor_b"]), decisions={"ss_type": 2, "include_interaction": value})
    with pytest.raises(SpecError, match="JSON boolean"):
        run(s, clinical)


def test_integer_decision_accepts_whole_floats_but_not_fractions(clinical):
    base = Spec(test="factorial_anova", variables=V(outcome="y", predictors=["group", "factor_b"]), decisions={"ss_type": 2.0, "include_interaction": True})
    r = run(base, clinical)
    assert r.extra["decisions"] == {"ss_type": 2.0, "include_interaction": True}, "decisions are echoed as written, never coerced"
    with pytest.raises(SpecError, match="integer"):
        run(base.model_copy(update={"decisions": {"ss_type": 2.5, "include_interaction": True}}), clinical)
    with pytest.raises(SpecError, match="one of \\[2, 3\\]"):
        run(base.model_copy(update={"decisions": {"ss_type": 1, "include_interaction": True}}), clinical)


def test_working_correlation_aliases_still_accepted(longitudinal):
    s = Spec(test="gee_cont", variables=V(outcome="y", predictors=["visit"], subject="id", time="visit"), decisions={"working_correlation": "AR(1)"})
    r = run(s, longitudinal)
    assert r.extra["decisions"] == {"working_correlation": "AR(1)"}


def test_factor_rotation_must_be_explicit(clinical):
    s = Spec(test="factor_sem", variables=V(columns=[f"item{i}" for i in range(6)]), decisions={"approach": "exploratory", "n_factors": 2})
    with pytest.raises(SpecError, match="rotation"):
        run(s, clinical)
    # and the conditional keys are not demanded for the other branch
    _, proc = preflight(Spec(test="factor_sem", variables=V(columns=["a"]), decisions={"approach": "confirmatory", "model_syntax": "F =~ a"}))
    assert [d.key for d in proc.decisions if d.when] == ["n_factors", "rotation", "model_syntax"]


def test_preflight_does_not_need_data():
    s = Spec(test="icc", variables=V(columns=["a", "b"]), decisions={"icc_form": "ICC(2,1)"})
    spec, proc = preflight(s)
    assert spec.test == "icc" and proc.test_id == "icc"
    with pytest.raises(SpecError, match="icc_form"):
        preflight(Spec(test="icc", variables=V(columns=["a", "b"]), decisions={"icc_form": "ICC9"}))
    with pytest.raises(SpecError, match="needs variables"):
        preflight(Spec(test="icc", decisions={"icc_form": "ICC2"}))


def test_cli_check_reports_missing_variables(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"test": "cox_cluster", "variables": {"time": "t", "event": "e", "predictors": ["x"]}, "decisions": {"approach": "stratify"}}))
    assert main(["check", str(p)]) == 1
    assert "INVALID" in capsys.readouterr().out
    p.write_text(json.dumps({"test": "cox_cluster", "variables": {"time": "t", "event": "e", "predictors": ["x"], "strata": "s"}, "decisions": {"approach": "stratify"}}))
    assert main(["check", str(p)]) == 0
    assert "approach='stratify'" in capsys.readouterr().out


def test_formula_errors_become_spec_errors(longitudinal, clinical):
    with pytest.raises(SpecError, match="formula error"):
        run(Spec(test="lmm", variables=V(outcome="y", predictors=["visit"], subject="id"), decisions={"re_formula": "~nope", "reml": True}), longitudinal)
    with pytest.raises(SpecError, match="formula error"):
        run(Spec(test="zip", variables=V(outcome="zcnt", predictors=["group"]), decisions={"count_distribution": "poisson", "inflation_formula": "nope"}), clinical)
    with pytest.raises(SpecError, match="formula error"):
        run(Spec(test="multiple_reg", variables=V(formula="y ~ group + nope")), clinical)


def test_cluster_robust_cox_headline_uses_robust_covariance(clinical):
    import statsmodels.formula.api as smf
    from scipy import stats

    v = V(time="time", event="event", predictors=["age", "group"], cluster="site")
    r = run(Spec(test="cox_cluster", variables=v, decisions={"approach": "cluster_robust"}), clinical)
    assert r.statistic_name.startswith("Wald") and "robust" in r.statistic_name
    assert r.p != pytest.approx(r.extra["lr_p"]), "the headline must not be the independence-based LR test"
    d = clinical.dropna(subset=["time", "event", "age", "group", "site"])
    ref = smf.phreg("time ~ age + C(group)", d, status=d.event.to_numpy(float), ties="efron").fit(groups=d.site.to_numpy())
    b = np.asarray(ref.params)
    W = float(b @ np.linalg.solve(np.asarray(ref.cov_params()), b))
    assert r.statistic == pytest.approx(W, rel=1e-9)
    assert r.p == pytest.approx(stats.chi2.sf(W, 2), rel=1e-9)
    assert r.df == 2
    strat = run(Spec(test="cox_cluster", variables=V(time="time", event="event", predictors=["age", "group"], strata="site"), decisions={"approach": "stratify"}), clinical)
    assert strat.statistic_name.startswith("LR")


def test_lmm_singular_first_optimizer_falls_back(longitudinal):
    rng = np.random.default_rng(9031)
    m, k = 60, 5
    t = np.tile(np.arange(-2.0, 3.0), m)
    d = pd.DataFrame({"id": np.repeat(np.arange(m), k), "t": t})
    d["y"] = 2 + 0.4 * t + np.repeat(rng.normal(0, 1.5, m), k) + rng.normal(0, 0.5, len(d)) + np.repeat(rng.normal(0, 0.9, m), k) * t
    r = run(Spec(test="lmm", variables=V(outcome="y", predictors=["t"], subject="id"), decisions={"re_formula": "1", "reml": True}), d)
    assert r.extra["optimizer"] in ("powell", "cg")
    assert any("optimizer fallback" in w for w in r.warnings)
    assert all(np.isfinite(e.value) for e in r.estimates)


def test_outlier_check_never_lists_observations():
    d = pd.DataFrame({"y": [*np.linspace(-1, 1, 20), 123456.789]})
    r = run(Spec(test="one_sample_t", variables=V(outcome="y")), d)
    text = r.summary() + r.to_json()
    assert "123456" not in text
    c = next(c for c in r.checks if c.key == "outliers")
    assert c.passed is False and "1 value(s)" in c.detail


def test_roc_extra_does_not_echo_score_values(clinical):
    r = run(Spec(test="roc_auc", variables=V(outcome="ybin", x="score") if "score" in clinical else V(outcome="event", x="age")), clinical)
    assert "thresholds" not in r.extra["roc"]


def test_binom_test_uses_positive_label():
    d = pd.DataFrame({"response": ["yes"] * 7 + ["no"] * 5 + [None]})
    r = run(Spec(test="binom_test", variables=V(outcome="response"), params={"positive_label": "yes"}), d)
    assert r.statistic == 7 and r.n == 12
    with pytest.raises(SpecError, match="positive_label"):
        run(Spec(test="binom_test", variables=V(outcome="response")), d)


@pytest.mark.parametrize("spec", [
    Spec(test="mcnemar", params={"table": [[20, 10], [30, 40]], "exact": "false"}),
    Spec(test="two_by_two", params={"table": [[20, 10], [30, 40]], "exact": 1}),
    Spec(test="binom_test", params={"k": 7, "n": 12, "p0": "bad"}),
    Spec(test="binom_test", params={"k": 7, "n": 12, "p0": 1.0}),
    Spec(test="fleiss_kappa", variables=V(columns=["r1", "r2", "r3"]), params={"n_boot": "many"}),
])
def test_typed_params_reject_lookalikes(spec, clinical):
    with pytest.raises(SpecError, match="params\\."):
        run(spec, clinical if spec.test == "fleiss_kappa" else None)
