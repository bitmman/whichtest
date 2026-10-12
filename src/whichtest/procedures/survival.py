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


# ---------------------------------------------------------------------------
# Cox proportional hazards (helper)
# ---------------------------------------------------------------------------
import statsmodels.formula.api as smf  # noqa: E402

from ..result import Estimate as _Est  # noqa: E402


@procedure("cox", requires=("time", "event", "predictors|formula"))
def cox(df: pd.DataFrame, spec: Spec) -> TestResult:
    import patsy

    v = spec.variables
    if v.formula:
        rhs = v.formula.split("~", 1)[1].strip()
    else:
        cat = set(v.categorical)
        parts = []
        for p in v.predictors:
            if p not in df.columns:
                raise SpecError(f"column {p!r} is not in the data")
            parts.append(f"C({p})" if (p in cat or not pd.api.types.is_numeric_dtype(df[p])) else p)
        rhs = " + ".join(parts)
    d = df.dropna(subset=[v.time, v.event, *v.predictors]).copy()
    e = column(d, v.event, "event")
    if not set(np.unique(e)) <= {0.0, 1.0}:
        raise SpecError("event column must be 0/1 (1 = event observed)")
    res = smf.phreg(f"{v.time} ~ {rhs}", d, status=d[v.event].to_numpy(), ties="efron").fit()
    names = list(res.model.exog_names)
    cc = None
    ph_check = checks.manual("proportional_hazards", "Proportional hazards: plot Schoenfeld residuals against time", "lifelines not installed")
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test

        X = patsy.dmatrix(rhs, d, return_type="dataframe").drop(columns="Intercept", errors="ignore")
        ld = X.copy()
        ld["_time"], ld["_event"] = d.loc[X.index, v.time].to_numpy(float), d.loc[X.index, v.event].to_numpy(float)
        cph = CoxPHFitter().fit(ld, duration_col="_time", event_col="_event")
        cc = compare(
            "lifelines.CoxPHFitter (Efron ties)",
            "lifelines uses Efron ties by default; statsmodels uses Breslow. Set ties='efron' or the two won't match.",
            {**{f"coef[{nm}]": float(b) for nm, b in zip(names, res.params)}, **{f"se[{nm}]": float(s) for nm, s in zip(names, res.bse)}},
            {**{f"coef[{nm}]": float(cph.params_[nm]) for nm in names}, **{f"se[{nm}]": float(cph.standard_errors_[nm]) for nm in names}},
            rtol=1e-4,
            atol=1e-6,
        )
        pht = proportional_hazard_test(cph, ld, time_transform="rank").summary
        worst = pht["p"].min()
        ph_check = CheckResult("proportional_hazards", "Proportional hazards (scaled Schoenfeld residuals vs rank time)", bool(worst >= spec.alpha), float(pht.loc[pht["p"].idxmin(), "test_statistic"]), float(worst), "; ".join(f"{nm}: p = {p:.3g}" for nm, p in pht["p"].items()))
    except ImportError:
        pass
    n_events = int(e.sum())
    epv = n_events / len(names) if names else float("inf")
    chk = [
        ph_check,
        CheckResult("events_per_variable", "Roughly 10 or more events per parameter", bool(epv >= 10), float(epv), None, f"{n_events} events / {len(names)} parameters = {epv:.1f}"),
        checks.manual("covariate_linearity", "Continuous covariates entered in a sensible functional form", "check martingale residuals or try splines"),
    ]
    warnings = []
    if ph_check.passed is False:
        warnings.append("Proportional hazards doubtful for at least one covariate: stratify on it or model a time-varying effect (cox_cluster).")
    ci = res.conf_int(alpha=1 - spec.ci_level)
    ests = [_Est(f"HR {nm}", float(np.exp(b)), (float(np.exp(ci[i, 0])), float(np.exp(ci[i, 1]))), spec.ci_level, f"p = {float(res.pvalues[i]):.3g}") for i, (nm, b) in enumerate(zip(names, res.params))]
    ll0 = res.model.loglike(np.zeros(len(names)))
    lr = float(2 * (res.llf - ll0))
    from scipy import stats as _st

    return TestResult(
        test="cox",
        n={"subjects": int(len(d)), "events": n_events},
        statistic_name="LR χ² (model)",
        statistic=lr,
        p=float(_st.chi2.sf(lr, len(names))),
        df=float(len(names)),
        estimates=ests,
        checks=chk,
        cross_check=cc,
        warnings=warnings,
        extra={"formula": f"{v.time} ~ {rhs}", "ties": "efron", "log_hr": {nm: float(b) for nm, b in zip(names, res.params)}},
    )
