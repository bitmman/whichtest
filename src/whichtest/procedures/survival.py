"""Time-to-event: Kaplan–Meier curves with the log-rank test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from .. import checks
from ..data import column
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


@procedure("km_logrank", requires=("time", "event", "group"))
def km_logrank(df: pd.DataFrame, spec: Spec) -> TestResult:
    v = spec.variables
    t = column(df, v.time, "time")
    e = column(df, v.event, "event")
    g = column(df, v.group, "group", numeric=False)
    keep = ~(np.isnan(t) | np.isnan(e)) & ~pd.isna(g)
    t, e, g = t[keep], e[keep], g[keep]
    if not set(np.unique(e)) <= {0.0, 1.0}:
        raise SpecError("event column must be 0/1 (1 = event observed)")
    labels = [l for l in pd.unique(g)]
    if len(labels) < 2:
        raise SpecError("km_logrank needs at least two groups")
    chisq, p = survdiff(t, e, g)
    cc = None
    try:
        from lifelines.statistics import logrank_test, multivariate_logrank_test

        if len(labels) == 2:
            a, b = g == labels[0], g == labels[1]
            lr = logrank_test(t[a], t[b], event_observed_A=e[a], event_observed_B=e[b])
        else:
            lr = multivariate_logrank_test(t, g.astype(str), e)
        cc = compare(
            "lifelines.statistics.logrank_test",
            "Second implementation of the same test. The chi-square should match.",
            {"chi2": chisq, "p": p},
            {"chi2": lr.test_statistic, "p": lr.p_value},
            rtol=1e-5,
        )
    except ImportError:
        pass
    ests = []
    timepoints = spec.params.get("timepoints", [])
    n = {}
    for lab in labels:
        mask = g == lab
        sf = SurvfuncRight(t[mask], e[mask])
        n[str(lab)] = int(mask.sum())
        med = sf.quantile(0.5)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = sf.quantile_ci(0.5, alpha=1 - spec.ci_level)
            ci = (float(lo), float(hi)) if np.isfinite(lo) and np.isfinite(hi) else None
        except Exception:
            ci = None
        ests.append(Estimate(f"median survival[{lab}]", None if (med is None or not np.isfinite(med)) else float(med), ci, spec.ci_level if ci else None, f"{int(e[mask].sum())} events / {int(mask.sum())}"))
        for tp in timepoints:
            s = float(np.interp(tp, sf.surv_times, sf.surv_prob, left=1.0)) if len(sf.surv_times) else 1.0
            ests.append(Estimate(f"S({tp:g})[{lab}]", s))
    # crossing curves: compare the ordering of the two curves early vs late
    crossing = checks.manual("crossing_curves", "Survival curves don't cross (log-rank loses power if they do)", "plot the curves")
    if len(labels) == 2:
        sfs = [SurvfuncRight(t[g == l], e[g == l]) for l in labels]
        grid = np.unique(t)
        diffs = np.interp(grid, sfs[0].surv_times, sfs[0].surv_prob, left=1.0) - np.interp(grid, sfs[1].surv_times, sfs[1].surv_prob, left=1.0)
        # a crossing worth worrying about: the curves separate by > 0.1 in both directions
        margin = float(spec.params.get("crossing_margin", 0.1))
        crosses = bool(diffs.max() > margin and diffs.min() < -margin)
        crossing = CheckResult("crossing_curves", "Survival curves don't cross", not crosses, float(max(diffs.max(), -diffs.min())), None, f"S[{labels[0]}] − S[{labels[1]}] ranges from {diffs.min():+.2f} to {diffs.max():+.2f} (margin {margin:g})")
    warnings = []
    if crossing.passed is False:
        warnings.append("Curves cross: consider restricted mean survival time (lifelines restricted_mean_survival_time).")
    return TestResult(
        test="km_logrank",
        n=n,
        statistic_name="log-rank χ²",
        statistic=float(chisq),
        p=float(p),
        df=float(len(labels) - 1),
        estimates=ests,
        checks=[checks.manual("noninformative_censoring", "Censoring unrelated to prognosis"), crossing],
        cross_check=cc,
        warnings=warnings,
    )
