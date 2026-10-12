"""Reusable assumption checks. Each returns a :class:`CheckResult`.

Checks never raise on small or degenerate samples; they report ``passed=None``
with a reason instead, so the test can still run and the human sees why.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from .result import CheckResult

def pg_col(frame, *names: str) -> str:
    """The first column of a pingouin frame matching any of ``names`` after normalising '-'/'_' and '%'."""
    norm = {c.replace("-", "_").rstrip("%"): c for c in frame.columns}
    for n in names:
        key = n.replace("-", "_").rstrip("%")
        if key in norm:
            return norm[key]
    raise KeyError(f"none of {names} in {list(frame.columns)}")


__all__ = [
    "shapiro",
    "shapiro_groups",
    "levene",
    "outliers_z",
    "expected_counts",
    "breusch_pagan",
    "vif",
    "overdispersion",
    "manual",
    "note",
]


def manual(key: str, text: str, detail: str = "") -> CheckResult:
    """A design property that only the human can confirm."""
    return CheckResult(key=key, text=text, passed=None, detail=detail or "confirm from the study design")


def note(key: str, text: str, detail: str, **data: Any) -> CheckResult:
    """An informational check that reports but cannot fail."""
    return CheckResult(key=key, text=text, passed=None, detail=detail, data=data)


def shapiro(x: np.ndarray, key: str, text: str, alpha: float = 0.05) -> CheckResult:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return CheckResult(key, text, None, detail="fewer than 3 values")
    if np.ptp(x) == 0:
        return CheckResult(key, text, None, detail="all values identical")
    if len(x) > 5000:
        x = np.random.default_rng(0).choice(x, 5000, replace=False)
    w, p = stats.shapiro(x)
    sk = float(stats.skew(x)) if len(x) > 2 else float("nan")
    detail = f"Shapiro–Wilk W = {w:.3f}; skew = {sk:.2f}; n = {len(x)}"
    if len(x) >= 300:
        detail += " (large n: Shapiro rejects tiny deviations; judge by the skew and a plot)"
    return CheckResult(key, text, bool(p >= alpha), float(w), float(p), detail, {"skew": sk, "n": int(len(x))})


def shapiro_groups(groups: dict[Any, np.ndarray], key: str, text: str, alpha: float = 0.05) -> CheckResult:
    parts, worst_p, anyfail, undecided = [], 1.0, False, False
    for label, g in groups.items():
        r = shapiro(g, key, text, alpha)
        if r.passed is None:
            undecided = True
            parts.append(f"{label}: {r.detail}")
        else:
            anyfail |= not r.passed
            worst_p = min(worst_p, r.p or 1.0)
            parts.append(f"{label}: W = {r.statistic:.3f}, p = {r.p:.3g}")
    passed = None if (undecided and not anyfail) else (not anyfail)
    return CheckResult(key, text, passed, None, worst_p if not undecided else None, "; ".join(parts))


def levene(groups: Sequence[np.ndarray], key: str, text: str, alpha: float = 0.05, informational: bool = False) -> CheckResult:
    gs = [np.asarray(g, float) for g in groups]
    gs = [g[~np.isnan(g)] for g in gs]
    if any(len(g) < 2 for g in gs):
        return CheckResult(key, text, None, detail="a group has fewer than 2 values")
    w, p = stats.levene(*gs, center="median")
    sds = ", ".join(f"{np.std(g, ddof=1):.3g}" for g in gs)
    detail = f"Levene (median-centred) W = {w:.3f}; group SDs: {sds}"
    passed = None if informational else bool(p >= alpha)
    return CheckResult(key, text, passed, float(w), float(p), detail)


def outliers_z(x: np.ndarray, key: str, text: str, z: float = 3.0) -> CheckResult:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return CheckResult(key, text, None, detail="too few values to assess")
    zs = np.abs(stats.zscore(x, ddof=1))
    idx = np.flatnonzero(zs > z)
    detail = f"{len(idx)} value(s) with |z| > {z:g}" + (f": {np.round(x[idx], 3).tolist()}" if len(idx) else "")
    return CheckResult(key, text, bool(len(idx) == 0), float(zs.max()), None, detail, {"n_outliers": int(len(idx))})


def expected_counts(table: np.ndarray, key: str, text: str) -> CheckResult:
    table = np.asarray(table, float)
    if table.sum() == 0:
        return CheckResult(key, text, None, detail="empty table")
    exp = stats.contingency.expected_freq(table)
    frac_small = float((exp < 5).mean())
    detail = f"min expected = {exp.min():.2f}; {frac_small:.0%} of cells < 5"
    ok = exp.min() >= 1 and frac_small <= 0.2
    return CheckResult(key, text, bool(ok), float(exp.min()), None, detail, {"expected": exp.tolist()})


def breusch_pagan(resid: np.ndarray, exog: np.ndarray, key: str, text: str, alpha: float = 0.05) -> CheckResult:
    from statsmodels.stats.diagnostic import het_breuschpagan

    if len(resid) <= exog.shape[1] + 2:
        return CheckResult(key, text, None, detail="too few residuals")
    lm, lm_p, f, f_p = het_breuschpagan(resid, exog)
    return CheckResult(key, text, bool(lm_p >= alpha), float(lm), float(lm_p), f"Breusch–Pagan LM = {lm:.3f}")


def vif(exog: np.ndarray, names: Sequence[str], key: str, text: str, threshold: float = 5.0) -> CheckResult:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = [i for i, n in enumerate(names) if n != "Intercept"]
    if len(cols) < 2:
        return CheckResult(key, text, None, detail="only one predictor; VIF not applicable")
    vals = {}
    for i in cols:
        try:
            vals[names[i]] = float(variance_inflation_factor(exog, i))
        except Exception:  # pragma: no cover
            vals[names[i]] = float("nan")
    worst = max(vals.values())
    detail = "; ".join(f"{k}: {v:.2f}" for k, v in vals.items())
    return CheckResult(key, text, bool(worst < threshold), worst, None, f"VIF {detail}", {"vif": vals})


def overdispersion(pearson_chi2: float, df_resid: float, key: str, text: str, threshold: float = 1.5) -> CheckResult:
    if df_resid <= 0:
        return CheckResult(key, text, None, detail="no residual degrees of freedom")
    phi = float(pearson_chi2 / df_resid)
    return CheckResult(key, text, bool(phi < threshold), phi, None, f"dispersion φ = Pearson χ² / df = {phi:.2f}")
