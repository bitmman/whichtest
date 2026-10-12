"""Count outcomes: two-sample rate comparison, Poisson / negative binomial regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.rates import confint_poisson_2indep, test_poisson_2indep

from .. import checks
from ..data import column
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError
from .regression import _coef_estimates, _formula


@procedure("poisson_2indep", data_optional=True)
def poisson_2indep(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    p = spec.params
    v = spec.variables
    if all(k in p for k in ("count1", "exposure1", "count2", "exposure2")):
        c1, e1, c2, e2 = int(p["count1"]), float(p["exposure1"]), int(p["count2"]), float(p["exposure2"])
        labels = ("1", "2")
    elif df is not None and v.outcome and v.group:
        y = column(df, v.outcome, "outcome")
        g = column(df, v.group, "group", numeric=False)
        ex = column(df, v.exposure, "exposure") if v.exposure else np.ones(len(y))
        labs = [l for l in pd.unique(g) if not pd.isna(l)]
        if len(labs) != 2:
            raise SpecError(f"poisson_2indep needs exactly two groups; found {labs}")
        c1, e1 = int(np.nansum(y[g == labs[0]])), float(np.nansum(ex[g == labs[0]]))
        c2, e2 = int(np.nansum(y[g == labs[1]])), float(np.nansum(ex[g == labs[1]]))
        labels = (str(labs[0]), str(labs[1]))
    else:
        raise SpecError("poisson_2indep needs params.count1/exposure1/count2/exposure2, or variables.outcome + group (+ exposure)")
    alt = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}[spec.alternative]
    r = test_poisson_2indep(c1, e1, c2, e2, method="score", alternative=alt)
    r2 = test_poisson_2indep(c1, e1, c2, e2, method="exact-cond", alternative=alt)
    cc = compare(
        "test_poisson_2indep(method='exact-cond')",
        "The conditional exact test should agree closely unless counts are small.",
        {"p": r.pvalue},
        {"p": r2.pvalue},
        kind="approximate",
        alpha=spec.alpha,
    )
    lo, hi = confint_poisson_2indep(c1, e1, c2, e2, method="score", alpha=1 - spec.ci_level)
    rate1, rate2 = c1 / e1, c2 / e2
    return TestResult(
        test="poisson_2indep",
        n={labels[0]: c1, labels[1]: c2},
        statistic_name="score z",
        statistic=float(r.statistic),
        p=float(r.pvalue),
        estimates=[
            Estimate(f"rate[{labels[0]}]", rate1, note=f"{c1} events / {e1:g} exposure"),
            Estimate(f"rate[{labels[1]}]", rate2, note=f"{c2} events / {e2:g} exposure"),
        ],
        effect_size=Estimate(f"rate ratio ({labels[0]} / {labels[1]})", float(rate1 / rate2) if rate2 > 0 else None, (float(lo), float(hi)), spec.ci_level),
        checks=[checks.manual("event_independence", "Events occur independently at a constant rate within each group")],
        cross_check=cc,
    )


@procedure("poisson_reg", requires=("outcome|formula",))
def poisson_reg(df: pd.DataFrame, spec: Spec) -> TestResult:
    f = _formula(spec, df)
    v = spec.variables
    offset = np.log(column(df, v.exposure, "exposure")) if v.exposure else None
    kw = {"offset": offset} if offset is not None else {}
    m = smf.glm(f, df, family=sm.families.Poisson(), **kw).fit()
    qp = smf.glm(f, df, family=sm.families.Poisson(), **kw).fit(scale="X2")
    disp = checks.overdispersion(float(m.pearson_chi2), float(m.df_resid), "overdispersion", "Variance ≈ mean (dispersion near 1)")
    y = np.asarray(m.model.endog)
    mu = np.asarray(m.predict())
    obs0, exp0 = float((y == 0).mean()), float(np.exp(-mu).mean())
    zeros = CheckResult("excess_zeros", "No more zeros than the model predicts", bool(obs0 <= exp0 * 1.5 + 0.02), obs0, None, f"observed zero fraction {obs0:.3f} vs Poisson-predicted {exp0:.3f}")
    warnings = []
    primary = m
    ests = _coef_estimates(m, spec.ci_level, exp=True, name="RR ")
    label = "Poisson"
    extra = {"formula": f, "dispersion": disp.statistic, "poisson_aic": float(m.aic)}
    if disp.passed is False:
        nb_kw = {"exposure": column(df, v.exposure, "exposure")} if v.exposure else {}
        nb = smf.negativebinomial(f, df, **nb_kw).fit(disp=0, maxiter=500)
        primary, label = nb, "negative binomial"
        ests = _coef_estimates(nb, spec.ci_level, exp=True, name="RR ")
        extra["nb_alpha"] = float(nb.params.get("alpha", float("nan")))
        extra["nb_aic"] = float(nb.aic)
        warnings.append(f"Overdispersed (φ = {disp.statistic:.2f}): switched to negative binomial regression. Poisson estimates kept in extra.")
        extra["poisson_coef"] = {n: float(b) for n, b in zip(m.params.index, m.params)}
        cc = compare(
            "quasi-Poisson GLM (scale='X2') p-values vs negative binomial",
            "If negative binomial and quasi-Poisson disagree sharply on inference, look at the mean–variance relationship.",
            {f"p[{n}]": float(p) for n, p in zip(nb.params.index, nb.pvalues) if n != "alpha"},
            {f"p[{n}]": float(p) for n, p in zip(qp.params.index, qp.pvalues)},
            kind="approximate",
            alpha=spec.alpha,
            p_tol=0.05,
        )
    else:
        cc = compare(
            "quasi-Poisson GLM (scale='X2')",
            "Quasi-Poisson has the same point estimates with SEs scaled by the dispersion; with no overdispersion the inference should match closely.",
            {**{f"b[{n}]": float(b) for n, b in zip(m.params.index, m.params)}, **{f"p[{n}]": float(p) for n, p in zip(m.params.index, m.pvalues)}},
            {**{f"b[{n}]": float(b) for n, b in zip(qp.params.index, qp.params)}, **{f"p[{n}]": float(p) for n, p in zip(qp.params.index, qp.pvalues)}},
            kind="approximate",
            alpha=spec.alpha,
            p_tol=0.05,
        )
    if zeros.passed is False:
        warnings.append("Excess zeros: consider a zero-inflated model (test id 'zip').")
    llnull = getattr(primary, "llnull", None)
    lr = float(2 * (primary.llf - llnull)) if llnull is not None else None
    lr_p = float(stats.chi2.sf(lr, primary.df_model)) if lr is not None else None
    return TestResult(
        test="poisson_reg",
        name=f"{label} regression",
        n=int(primary.nobs),
        statistic_name="LR χ² (model)" if lr is not None else "",
        statistic=lr,
        p=lr_p,
        df=float(primary.df_model),
        estimates=ests,
        checks=[disp, zeros],
        cross_check=cc,
        warnings=warnings,
        extra=extra,
    )
