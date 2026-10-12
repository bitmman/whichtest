"""Boundary contracts through public APIs; failures are intentional review targets.

These are ordinary tests, not xfails. Fix the implementation or explicitly
justify a contract change; do not bless current erroneous output as expected.
"""
import builtins
import json

import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, SpecError, run
from whichtest.cli import main
from whichtest.result import TestResult as PipelineResult, compare


@pytest.mark.parametrize("field,bad", [("k", 2.8), ("k", -.5), ("n", 10.5)])
def test_binomial_rejects_fractional_summary_counts(field, bad):
    params = {"k": 2, "n": 10, field: bad}
    with pytest.raises(SpecError):
        run(Spec(test="binom_test", params=params))


@pytest.mark.parametrize("field,bad", [
    ("count1", 2.8), ("count2", -.5), ("count1", np.inf),
    ("exposure1", np.nan), ("exposure2", np.inf), ("exposure1", 0),
])
def test_poisson_summary_requires_finite_integer_counts_and_finite_exposure(field, bad):
    params = {"count1": 10, "count2": 12, "exposure1": 3, "exposure2": 4, field: bad}
    with pytest.raises(SpecError):
        run(Spec(test="poisson_2indep", params=params))


@pytest.mark.parametrize("test", ["two_by_two", "chi_sq_indep", "mcnemar", "stuart_maxwell"])
def test_contingency_tables_reject_fractional_counts(test):
    with pytest.raises(SpecError):
        run(Spec(test=test, params={"table": [[10.5, 6], [7, 12]]}))


@pytest.mark.parametrize("code", [1.9, "1.9"])
def test_competing_event_of_interest_is_not_truncated(code):
    d = pd.DataFrame({"time": [1,2,3,4,5,6], "event": [1,2,0,1,2,0]})
    with pytest.raises(SpecError):
        run(Spec(test="competing", variables={"time": "time", "event": "event"},
                 decisions={"estimand": "cumulative_incidence", "event_of_interest": code}), d)


@pytest.mark.parametrize("test", ["cohen_kappa", "fleiss_kappa", "stuart_maxwell"])
def test_nominal_category_renaming_does_not_change_analysis(test):
    d = pd.DataFrame({"a": [0,0,0,1,1,1,2,2,2,2],
                      "b": [0,1,0,1,2,1,2,0,2,1],
                      "c": [0,0,1,1,2,0,2,2,2,1]})
    columns = ["a", "b", "c"] if test == "fleiss_kappa" else ["a", "b"]
    spec = Spec(test=test, variables={"columns": columns}, params={"n_boot": 80, "seed": 51})
    base = run(spec, d)
    renamed = run(spec, d.replace({0: "absent", 1: "possible", 2: "present"}))
    assert renamed.n == base.n
    if base.p is not None:
        assert renamed.p == pytest.approx(base.p)
    if base.effect_size is not None:
        assert renamed.effect_size.value == pytest.approx(base.effect_size.value)


def test_dense_two_by_two_does_not_silently_ignore_direction():
    params = {"table": [[30,20], [20,30]]}  # odds ratio > 1, all expectations >=5
    try:
        less = run(Spec(test="two_by_two", params=params, alternative="less"))
        greater = run(Spec(test="two_by_two", params=params, alternative="greater"))
    except SpecError:
        return  # Rejecting unsupported directional chi-square is acceptable.
    assert greater.p < .5 < less.p, "opposite hypotheses cannot both receive the two-sided chi-square p"


@pytest.mark.parametrize("kind,quantity", [("exact", "x"), ("approximate", "p"), ("direction", "direction")])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_comparisons_never_verify(kind, quantity, bad):
    for a, b in [(bad, bad), (bad, .5), (.5, bad)]:
        cc = compare("oracle", "finite values required", {quantity: a}, {quantity: b}, kind=kind)
        assert not cc.agrees


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
def test_result_cannot_hide_nonfinite_inference_in_numpy_scalar(dtype):
    cc = compare("oracle", "known equal values", {"x": 1}, {"x": 1})
    r = PipelineResult(test="probe", statistic=dtype(np.nan), p=.5, cross_check=cc)
    assert r.inference_failed
    assert r.flagged
    assert json.loads(r.to_json())["statistic"] is None


def test_empty_comparison_does_not_verify_any_result():
    cc = compare("oracle", "no shared quantities", {}, {})
    assert not cc.agrees, "all([]) is not evidence that two results agree"


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
def test_nonfinite_pvalue_is_unavailable_and_flagged(dtype):
    r = PipelineResult(test="probe", p=dtype(np.nan), cross_check=compare("m", "w", {"x": 1}, {"x": 1}))
    assert r.significant is None
    assert r.flagged


def test_cli_strict_emits_json_and_reports_missing_cross_check(tmp_path, monkeypatch, capsys):
    # Infrastructure simulation only; numerical oracles elsewhere never mock a fit.
    original = builtins.__import__
    def without_pingouin(name, *args, **kwargs):
        if name == "pingouin" or name.startswith("pingouin."):
            raise ImportError("simulated missing optional dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_pingouin)
    spec = tmp_path / "spec.json"
    data = tmp_path / "data.csv"
    out = tmp_path / "result.json"
    spec.write_text(Spec(test="friedman", variables={"columns": ["a", "b", "c"]}).to_json())
    pd.DataFrame({"a": [1,2,3,5,6], "b": [2,3,5,6,7], "c": [3,4,6,7,8]}).to_csv(data, index=False)
    code = main(["run", str(spec), str(data), "--strict", "--json", "--out", str(out)])
    printed = json.loads(capsys.readouterr().out)
    stored = json.loads(out.read_text())
    assert code == 2
    assert printed == stored
    assert stored["flagged"] is True
    assert stored["cross_check"]["agrees"] is False
    assert "NOT RUN" in stored["cross_check"]["detail"]
