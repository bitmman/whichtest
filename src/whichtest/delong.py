"""DeLong's method for the variance of an AUC and the paired comparison of two AUCs.

Fast implementation after Sun & Xu (2014), "Fast implementation of DeLong's
algorithm for comparing the areas under correlated receiver operating
characteristic curves", IEEE Signal Processing Letters 21(11).

Nothing in scipy, statsmodels or scikit-learn provides this; it is written once
here and validated in ``tests/test_delong.py`` against scikit-learn's AUC and a
paired bootstrap.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["midrank", "delong_auc_variance", "delong_paired", "delong_ci"]


def midrank(x: np.ndarray) -> np.ndarray:
    """Average (mid) ranks, 1-based, ties share the mean rank."""
    return stats.rankdata(np.asarray(x, float), method="average")


def _fast_delong(preds: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """``preds``: k × n scores with the ``m`` positives first. Returns (aucs, covariance)."""
    k, n_total = preds.shape
    n = n_total - m
    tx = np.vstack([midrank(preds[r, :m]) for r in range(k)])
    ty = np.vstack([midrank(preds[r, m:]) for r in range(k)])
    tz = np.vstack([midrank(preds[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.atleast_2d(np.var(v01, ddof=1))
    sy = np.cov(v10) if k > 1 else np.atleast_2d(np.var(v10, ddof=1))
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def _order(y: np.ndarray, *scores: np.ndarray) -> tuple[np.ndarray, int]:
    y = np.asarray(y).astype(int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1 with 1 = positive")
    order = np.argsort(-y, kind="stable")
    preds = np.vstack([np.asarray(s, float)[order] for s in scores])
    return preds, int(y.sum())


def delong_auc_variance(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC and its DeLong variance for one score."""
    preds, m = _order(y, score)
    if m == 0 or m == preds.shape[1]:
        raise ValueError("need both positive and negative cases")
    aucs, cov = _fast_delong(preds, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_ci(y: np.ndarray, score: np.ndarray, level: float = 0.95) -> tuple[float, tuple[float, float], float]:
    auc, var = delong_auc_variance(y, score)
    z = stats.norm.ppf(0.5 + level / 2)
    se = np.sqrt(var)
    return auc, (float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))), float(se)


def delong_paired(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, level: float = 0.95) -> dict:
    """Compare two AUCs measured on the same cases.

    Returns auc_a, auc_b, diff, se, ci, z, p (two-sided).
    """
    preds, m = _order(y, score_a, score_b)
    if m == 0 or m == preds.shape[1]:
        raise ValueError("need both positive and negative cases")
    aucs, cov = _fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0:
        z, p = (0.0, 1.0) if diff == 0 else (float("inf") * np.sign(diff), 0.0)
    else:
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
    q = stats.norm.ppf(0.5 + level / 2)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": diff,
        "se": se,
        "ci": (diff - q * se, diff + q * se),
        "z": float(z),
        "p": p,
        "cov": cov,
    }
