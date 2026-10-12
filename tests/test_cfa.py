"""Confirmatory factor analysis (factor_sem, approach='confirmatory') via semopy."""
import numpy as np
import pandas as pd
import pytest

from whichtest import Spec, SpecError, run

semopy = pytest.importorskip("semopy")
V = dict


@pytest.fixture(scope="module")
def two_factor_items():
    rng = np.random.default_rng(0)
    n = 300
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    lam = [0.8, 0.7, 0.6, 0.8, 0.7, 0.6]
    return pd.DataFrame({f"i{i}": (f1 if i < 3 else f2) * lam[i] + rng.normal(0, 0.6, n) for i in range(6)})


def test_correct_structure_fits_and_matches_efa(two_factor_items):
    spec = Spec(test="factor_sem", variables=V(columns=[f"i{i}" for i in range(6)]), decisions={"approach": "confirmatory", "model_syntax": "F1 =~ i0 + i1 + i2\nF2 =~ i3 + i4 + i5"})
    r = run(spec, two_factor_items)
    assert r.cross_check.agrees, r.summary()
    fit = r.extra["fit"]
    assert fit["CFI"] > 0.95 and fit["RMSEA"] < 0.06
    assert fit["DoF"] == 8  # 21 moments − 13 free parameters
    assert all(c.passed for c in r.checks if c.key == "model_fit")
    # standardized loadings: each item loads on its own factor, not the other
    L = r.extra["standardized_loadings"]
    for i in range(6):
        own, other = ("F1", "F2") if i < 3 else ("F2", "F1")
        assert L[f"i{i}"][own] > 0.6 and L[f"i{i}"][other] == 0.0
    assert not r.flagged


def test_misspecified_structure_fails_fit_check(two_factor_items):
    # one factor forced onto a two-factor structure
    spec = Spec(test="factor_sem", variables=V(columns=[f"i{i}" for i in range(6)]), decisions={"approach": "confirmatory", "model_syntax": "G =~ i0 + i1 + i2 + i3 + i4 + i5"})
    r = run(spec, two_factor_items)
    fit_check = next(c for c in r.checks if c.key == "model_fit")
    assert fit_check.passed is False
    assert r.flagged


def test_confirmatory_requires_model_syntax(two_factor_items):
    with pytest.raises(SpecError, match="model_syntax"):
        run(Spec(test="factor_sem", variables=V(columns=[f"i{i}" for i in range(6)]), decisions={"approach": "confirmatory"}), two_factor_items)


def test_base_install_message_without_semopy(two_factor_items, monkeypatch):
    import builtins

    real = builtins.__import__

    def blocker(name, *a, **k):
        if name.startswith("semopy"):
            raise ImportError("semopy")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocker)
    with pytest.raises(SpecError, match="semopy"):
        run(Spec(test="factor_sem", variables=V(columns=[f"i{i}" for i in range(6)]), decisions={"approach": "confirmatory", "model_syntax": "F =~ i0 + i1"}), two_factor_items)
