"""Effect sizes and confidence intervals shared by procedures."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .result import Estimate

__all__ = [
    "mean_ci",
    "mean_diff_ci_welch",
    "cohen_d_one",
    "hedges_g",
    "d_z",
    "rank_biserial_paired",
    "rank_biserial_indep",
    "hodges_lehmann_one",
    "hodges_lehmann_paired",
    "eta_omega",
    "epsilon_sq",
    "cramers_v",
    "cohen_w",
    "kendall_w",
    "fisher_z_ci",
    "bootstrap_ci",
]


def _t(level: float, df: float) -> float:
    return float(stats.t.ppf(0.5 + level / 2, df))


def mean_ci(x: np.ndarray, level: float = 0.95, mu0: float = 0.0, name: str = "mean difference") -> Estimate:
    n = len(x)
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n))
    h = _t(level, n - 1) * se
    return Estimate(name, m - mu0, (m - mu0 - h, m - mu0 + h), level)


def mean_diff_ci_welch(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> Estimate:
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    d = float(np.mean(a) - np.mean(b))
    h = _t(level, df) * se
    return Estimate("mean difference", d, (d - h, d + h), level)


def cohen_d_one(x: np.ndarray, mu0: float) -> Estimate:
    sd = np.std(x, ddof=1)
    return Estimate("Cohen's d", float((np.mean(x) - mu0) / sd) if sd > 0 else None)


def hedges_g(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> Estimate:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return Estimate("Hedges' g", None)
    d = (np.mean(a) - np.mean(b)) / sp
    df = na + nb - 2
    j = 1 - 3 / (4 * df - 1)  # small-sample correction
    g = d * j
    se = np.sqrt((na + nb) / (na * nb) + g**2 / (2 * (na + nb)))
    z = stats.norm.ppf(0.5 + level / 2)
    return Estimate("Hedges' g", float(g), (float(g - z * se), float(g + z * se)), level, "large-sample CI")


def d_z(diff: np.ndarray) -> Estimate:
    sd = np.std(diff, ddof=1)
    return Estimate("d_z", float(np.mean(diff) / sd) if sd > 0 else None, note="mean(diff) / SD(diff)")


def rank_biserial_paired(diff: np.ndarray) -> Estimate:
    d = diff[diff != 0]
    if len(d) == 0:
        return Estimate("matched-pairs rank-biserial r", None)
    r = stats.rankdata(np.abs(d))
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return Estimate("matched-pairs rank-biserial r", float((rp - rn) / (rp + rn)))


def rank_biserial_indep(u: float, n1: int, n2: int) -> tuple[Estimate, Estimate]:
    cl = u / (n1 * n2)
    return (
        Estimate("rank-biserial r", float(2 * cl - 1), note="U-based; positive = first group larger"),
        Estimate("common-language effect size U/(n1 n2)", float(cl), note="P(first > second)"),
    )


def hodges_lehmann_one(x: np.ndarray, level: float = 0.95) -> Estimate:
    """Walsh-average pseudo-median with a distribution-free CI."""
    x = np.asarray(x, float)
    n = len(x)
    i, j = np.triu_indices(n)
    w = np.sort((x[i] + x[j]) / 2)
    est = float(np.median(w))
    ci = None
    if n >= 6:
        m = n * (n + 1) / 2
        mu, sd = m / 2, np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        k = int(np.floor(mu - stats.norm.ppf(0.5 + level / 2) * sd))
        k = max(k, 1)
        if k <= len(w) - k:
            ci = (float(w[k - 1]), float(w[len(w) - k]))
    return Estimate("Hodges–Lehmann pseudo-median", est, ci, level if ci else None)


def hodges_lehmann_paired(diff: np.ndarray, level: float = 0.95) -> Estimate:
    e = hodges_lehmann_one(diff, level)
    e.name = "Hodges–Lehmann median difference"
    return e


def eta_omega(ss_between: float, ss_within: float, df_between: float, df_within: float) -> tuple[Estimate, Estimate]:
    ss_total = ss_between + ss_within
    eta = ss_between / ss_total if ss_total > 0 else None
    ms_w = ss_within / df_within if df_within > 0 else None
    omega = None
    if ms_w is not None and ss_total + ms_w > 0:
        omega = max(0.0, (ss_between - df_between * ms_w) / (ss_total + ms_w))
    return Estimate("eta²", None if eta is None else float(eta)), Estimate("omega²", None if omega is None else float(omega))


def epsilon_sq(h: float, n: int) -> Estimate:
    return Estimate("epsilon²", float(h / (n - 1)) if n > 1 else None, note="H / (n − 1)")


def cramers_v(table: np.ndarray) -> Estimate:
    return Estimate("Cramér's V", float(stats.contingency.association(np.rint(np.asarray(table)).astype(int), method="cramer")))


def cohen_w(chi2: float, n: int) -> Estimate:
    return Estimate("Cohen's w", float(np.sqrt(chi2 / n)))


def kendall_w(chi2: float, n: int, k: int) -> Estimate:
    return Estimate("Kendall's W", float(chi2 / (n * (k - 1))))


def fisher_z_ci(r: float, n: int, level: float = 0.95, name: str = "r") -> Estimate:
    if n < 4 or abs(r) >= 1:
        return Estimate(name, float(r))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    h = stats.norm.ppf(0.5 + level / 2) * se
    return Estimate(name, float(r), (float(np.tanh(z - h)), float(np.tanh(z + h))), level, "Fisher z")


def bootstrap_ci(statistic, *arrays: np.ndarray, level: float = 0.95, n_boot: int = 2000, seed: int = 0, paired: bool = True) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(statistic(*(a[idx] for a in arrays)))
        except Exception:
            continue
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
