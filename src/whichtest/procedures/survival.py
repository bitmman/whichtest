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


def _surv_at(sf: SurvfuncRight, t: float) -> float:
    """Right-continuous Kaplan–Meier step function evaluated at ``t`` (1 before the first event)."""
    times = np.asarray(sf.surv_times, float)
    if len(times) == 0:
        return 1.0
    i = int(np.searchsorted(times, t, side="right")) - 1
    return 1.0 if i < 0 else float(sf.surv_prob[i])


def _surv_curve(sf: SurvfuncRight, grid: np.ndarray) -> np.ndarray:
    return np.array([_surv_at(sf, t) for t in grid])


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
            ests.append(Estimate(f"S({tp:g})[{lab}]", _surv_at(sf, float(tp)), note="Kaplan–Meier step function"))
    # crossing curves: compare the ordering of the two curves early vs late
    crossing = checks.manual("crossing_curves", "Survival curves don't cross (log-rank loses power if they do)", "plot the curves")
    if len(labels) == 2:
        sfs = [SurvfuncRight(t[g == l], e[g == l]) for l in labels]
        grid = np.unique(t)
        diffs = _surv_curve(sfs[0], grid) - _surv_curve(sfs[1], grid)
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


# ---------------------------------------------------------------------------
# Competing risks (judgment: estimand, event of interest)
# ---------------------------------------------------------------------------
@procedure("competing", requires=("time", "event"), decisions=("estimand", "event_of_interest"))
def competing(df: pd.DataFrame, spec: Spec) -> TestResult:
    v = spec.variables
    estimand = str(spec.decisions["estimand"]).lower()
    if estimand not in ("cumulative_incidence", "cause_specific_hazard"):
        raise SpecError("decisions.estimand must be 'cumulative_incidence' or 'cause_specific_hazard'")
    eoi = spec.decisions["event_of_interest"]
    d = df.dropna(subset=[v.time, v.event]).copy()
    t = d[v.time].to_numpy(float)
    # normalise event codes once: numeric codes compare as numbers (0 = censored), others as strings
    raw_et = d[v.event].to_numpy()
    numeric_codes = pd.api.types.is_numeric_dtype(d[v.event])
    if numeric_codes:
        et_num = raw_et.astype(float)
        if (et_num != np.round(et_num)).any():
            raise SpecError(f"{v.event!r} must hold integer event codes (0 = censored)")
        codes = et_num.astype(int)
        censor = 0
        try:
            eoi_f = float(eoi)
        except (TypeError, ValueError):
            raise SpecError(f"event_of_interest {eoi!r} is not a numeric code; {v.event!r} holds {np.unique(codes).tolist()}") from None
        if not np.isfinite(eoi_f) or eoi_f != np.round(eoi_f):
            raise SpecError(f"event_of_interest {eoi!r} is not an integer event code; {v.event!r} holds {np.unique(codes).tolist()}")
        eoi_code = int(eoi_f)
    else:
        codes = raw_et.astype(str)
        censor = str(spec.params.get("censor_code", "0"))
        eoi_code = str(eoi)
    if eoi_code == censor:
        raise SpecError(f"event_of_interest {eoi!r} is the censoring code")
    if eoi_code not in set(codes.tolist()):
        raise SpecError(f"event_of_interest {eoi!r} not among event codes {np.unique(codes).tolist()} ({censor!r} = censored)")
    is_eoi = (codes == eoi_code).astype(int)
    any_event = (codes != censor).astype(int)
    n_eoi, n_comp, n_cens = int(is_eoi.sum()), int(any_event.sum() - is_eoi.sum()), int((any_event == 0).sum())
    extra_events = {"of_interest": n_eoi, "competing": n_comp, "censored": n_cens}
    naive = SurvfuncRight(t, is_eoi)
    naive_ci_end = float(1 - naive.surv_prob[-1]) if len(naive.surv_prob) else 0.0
    timepoints = list(spec.params.get("timepoints", []))
    ests: list[Estimate] = []
    cc = None
    extra: dict = {"events": extra_events}
    try:
        from lifelines import AalenJohansenFitter

        if numeric_codes:
            aj_codes, aj_eoi = codes, eoi_code
        else:  # lifelines needs integer codes with 0 = censored
            others = sorted(set(codes.tolist()) - {censor})
            mapping = {censor: 0, **{c: i + 1 for i, c in enumerate(others)}}
            aj_codes, aj_eoi = np.array([mapping[c] for c in codes]), mapping[eoi_code]
        ajf = AalenJohansenFitter(alpha=1 - spec.ci_level, calculate_variance=True, seed=int(spec.params.get("seed", 0))).fit(t, aj_codes, event_of_interest=aj_eoi)
        cif = ajf.cumulative_density_.iloc[:, 0]
        aj_end = float(cif.iloc[-1])
        cc = compare(
            "1 − naive Kaplan–Meier treating competing events as censored",
            "One minus naive Kaplan–Meier overstates cumulative incidence. The size of the gap shows how much competing risks matter here.",
            {"cumulative_incidence_at_end": aj_end},
            {"cumulative_incidence_at_end": naive_ci_end},
            kind="exact",
            rtol=0.0,
            atol=float("inf"),
            detail=f"Aalen–Johansen {aj_end:.3f} vs 1 − KM {naive_ci_end:.3f}: gap = {naive_ci_end - aj_end:+.3f} (the naive estimate must be ≥; the gap is the competing-risk bias)",
        )
        cc.agrees = bool(naive_ci_end >= aj_end - 1e-9)
        ci_df = ajf.confidence_interval_
        for tp in timepoints:
            idx = cif.index.searchsorted(tp, side="right") - 1
            val = float(cif.iloc[idx]) if idx >= 0 else 0.0
            lo = float(ci_df.iloc[idx, 1]) if idx >= 0 else 0.0
            hi = float(ci_df.iloc[idx, 0]) if idx >= 0 else 0.0
            ests.append(Estimate(f"cumulative incidence of {eoi} at {tp:g}", val, (min(lo, hi), max(lo, hi)), spec.ci_level, "Aalen–Johansen"))
        ests.append(Estimate(f"cumulative incidence of {eoi} at end of follow-up", aj_end, note="Aalen–Johansen"))
        ests.append(Estimate("1 − KM (naive) at end of follow-up", naive_ci_end, note="overstates incidence"))
        extra["aalen_johansen"] = {"time": [float(x) for x in cif.index], "cif": [float(x) for x in cif.values]}
    except ImportError:
        if estimand == "cumulative_incidence":
            raise SpecError("cumulative incidence needs lifelines (Aalen–Johansen): pip install 'whichtest[cross]'") from None
        ests.append(Estimate("1 − KM (naive) at end of follow-up", naive_ci_end, note="lifelines not installed; no Aalen–Johansen cross-check"))
    stat_name, stat, p, dfree = "", None, None, None
    if estimand == "cause_specific_hazard":
        if not (v.predictors or v.formula):
            raise SpecError("cause_specific_hazard needs variables.predictors (or formula) for the Cox model")
        if v.formula:
            rhs = v.formula.split("~", 1)[1].strip()
        else:
            cat = set(v.categorical)
            rhs = " + ".join(f"C({p_})" if (p_ in cat or not pd.api.types.is_numeric_dtype(d[p_])) else p_ for p_ in v.predictors)
        dd = d.dropna(subset=list(v.predictors)).copy()
        status = is_eoi[d.index.get_indexer(dd.index)]
        res = smf.phreg(f"{v.time} ~ {rhs}", dd, status=status, ties="efron").fit()
        names = list(res.model.exog_names)
        ci = res.conf_int(alpha=1 - spec.ci_level)
        ests += [_Est(f"cause-specific HR {nm}", float(np.exp(b)), (float(np.exp(ci[i, 0])), float(np.exp(ci[i, 1]))), spec.ci_level, f"p = {float(res.pvalues[i]):.3g}") for i, (nm, b) in enumerate(zip(names, res.params))]
        from scipy import stats as _st

        lr = float(2 * (res.llf - res.model.loglike(np.zeros(len(names)))))
        stat_name, stat, p, dfree = "LR χ² (cause-specific Cox)", lr, float(_st.chi2.sf(lr, len(names))), float(len(names))
        extra["cause_specific_cox"] = {"formula": f"{v.time} ~ {rhs}", "competing events treated as": "censored"}
    chk = [checks.note("estimand", "Estimand chosen: cause-specific hazards for etiology, cumulative incidence for prognosis", f"{estimand}; event of interest = {eoi!r}; chosen by the user")]
    return TestResult(
        test="competing",
        n={"subjects": int(len(d)), "events_of_interest": n_eoi, "competing_events": n_comp, "censored": n_cens},
        statistic_name=stat_name,
        statistic=stat,
        p=p,
        df=dfree,
        estimates=ests,
        checks=chk,
        cross_check=cc,
        warnings=["Fine–Gray subdistribution hazards are not available in statsmodels or lifelines; cause-specific hazards are reported instead."] if estimand == "cause_specific_hazard" else [],
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Clustered / stratified Cox (judgment: approach)
# ---------------------------------------------------------------------------
@procedure("cox_cluster", requires=("time", "event", "predictors|formula", "cluster|strata"), decisions=("approach",))
def cox_cluster(df: pd.DataFrame, spec: Spec) -> TestResult:
    import patsy

    v = spec.variables
    approach = str(spec.decisions["approach"]).lower()
    if approach not in ("cluster_robust", "stratify"):
        raise SpecError("decisions.approach must be 'cluster_robust' or 'stratify'")
    if approach == "cluster_robust" and not v.cluster:
        raise SpecError("cluster_robust needs variables.cluster")
    if approach == "stratify" and not v.strata:
        raise SpecError("stratify needs variables.strata")
    if v.formula:
        rhs = v.formula.split("~", 1)[1].strip()
    else:
        cat = set(v.categorical)
        rhs = " + ".join(f"C({p})" if (p in cat or not pd.api.types.is_numeric_dtype(df[p])) else p for p in v.predictors)
    extra_col = v.cluster if approach == "cluster_robust" else v.strata
    d = df.dropna(subset=[v.time, v.event, extra_col, *v.predictors]).copy()
    status = d[v.event].to_numpy(float)
    if approach == "cluster_robust":
        res = smf.phreg(f"{v.time} ~ {rhs}", d, status=status, ties="efron").fit(groups=d[v.cluster].to_numpy())
    else:
        res = smf.phreg(f"{v.time} ~ {rhs}", d, status=status, strata=d[v.strata].to_numpy(), ties="efron").fit()
    names = list(res.model.exog_names)
    cc = None
    try:
        from lifelines import CoxPHFitter

        X = patsy.dmatrix(rhs, d, return_type="dataframe").drop(columns="Intercept", errors="ignore")
        ld = X.copy()
        ld["_time"], ld["_event"] = d.loc[X.index, v.time].to_numpy(float), status
        if approach == "cluster_robust":
            ld["_cluster"] = d.loc[X.index, v.cluster].to_numpy()
            cph = CoxPHFitter().fit(ld, duration_col="_time", event_col="_event", cluster_col="_cluster")
        else:
            ld["_strata"] = d.loc[X.index, v.strata].to_numpy()
            cph = CoxPHFitter().fit(ld, duration_col="_time", event_col="_event", strata=["_strata"])
        cc = compare(
            f"lifelines.CoxPHFitter ({'cluster_col' if approach == 'cluster_robust' else 'strata'})",
            "lifelines' cluster-robust standard errors should be close to statsmodels'.",
            {**{f"coef[{nm}]": float(b) for nm, b in zip(names, res.params)}, **{f"se[{nm}]": float(s) for nm, s in zip(names, res.bse)}},
            {**{f"coef[{nm}]": float(cph.params_[nm]) for nm in names}, **{f"se[{nm}]": float(cph.standard_errors_[nm]) for nm in names}},
            rtol=1e-3,
            atol=1e-6,
        )
    except ImportError:
        pass
    n_clusters = int(d[extra_col].nunique())
    chk = [
        CheckResult("cluster_count", "Enough clusters for robust standard errors (≥ 40)" if approach == "cluster_robust" else "Enough events within each stratum", bool(n_clusters >= 40) if approach == "cluster_robust" else bool(d.groupby(extra_col)[v.event].sum().min() >= 5), float(n_clusters), None, f"{n_clusters} {'clusters' if approach == 'cluster_robust' else 'strata'}" + ("" if approach == "cluster_robust" else f"; min events per stratum = {int(d.groupby(extra_col)[v.event].sum().min())}")),
        checks.note("frailty_vs_robust", "Robust SEs and frailty models answer different questions. Choose deliberately", f"approach = {approach}; chosen by the user"),
    ]
    ci = res.conf_int(alpha=1 - spec.ci_level)
    ests = [_Est(f"HR {nm}", float(np.exp(b)), (float(np.exp(ci[i, 0])), float(np.exp(ci[i, 1]))), spec.ci_level, f"p = {float(res.pvalues[i]):.3g}") for i, (nm, b) in enumerate(zip(names, res.params))]
    from scipy import stats as _st

    lr = float(2 * (res.llf - res.model.loglike(np.zeros(len(names)))))
    return TestResult(
        test="cox_cluster",
        n={"subjects": int(len(d)), "events": int(status.sum()), ("clusters" if approach == "cluster_robust" else "strata"): n_clusters},
        statistic_name="LR χ² (model)",
        statistic=lr,
        p=float(_st.chi2.sf(lr, len(names))),
        df=float(len(names)),
        estimates=ests,
        checks=chk,
        cross_check=cc,
        extra={"formula": f"{v.time} ~ {rhs}", "approach": approach, "by": extra_col},
    )
