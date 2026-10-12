"""Count outcomes: two-sample rate comparison, Poisson / negative binomial regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.rates import confint_poisson_2indep, test_poisson_2indep

from .. import checks
from ..data import column, int_param, positive_float_param
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError
from .regression import _coef_estimates, _formula


@procedure("poisson_2indep", data_optional=True, directional=True)
def poisson_2indep(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    p = spec.params
    v = spec.variables
    n_drop = 0
    if any(k in p for k in ("count1", "exposure1", "count2", "exposure2")):
        c1, c2 = int_param(p, "count1"), int_param(p, "count2")  # validated before any cast
        e1, e2 = positive_float_param(p, "exposure1"), positive_float_param(p, "exposure2")
        labels = ("1", "2")
    elif df is not None and v.outcome and v.group:
        y = column(df, v.outcome, "outcome")
        g = column(df, v.group, "group", numeric=False)
        ex = column(df, v.exposure, "exposure") if v.exposure else np.ones(len(y))
        keep = ~np.isnan(y) & ~np.isnan(ex) & ~pd.isna(g)  # one joint mask: a row with any missing part is out
        n_drop = int((~keep).sum())
        y, g, ex = y[keep], g[keep], ex[keep]
        if (y < 0).any() or (y != np.round(y)).any():
            raise SpecError(f"{v.outcome!r} must hold non-negative integer counts")
        if (ex <= 0).any() or not np.isfinite(ex).all():
            raise SpecError(f"{v.exposure!r} must hold positive finite exposures")
        labs = [l for l in pd.unique(g)]
        if len(labs) != 2:
            raise SpecError(f"poisson_2indep needs exactly two groups; found {labs}")
        c1, e1 = int(y[g == labs[0]].sum()), float(ex[g == labs[0]].sum())
        c2, e2 = int(y[g == labs[1]].sum()), float(ex[g == labs[1]].sum())
        labels = (str(labs[0]), str(labs[1]))
    else:
        raise SpecError("poisson_2indep needs params.count1/exposure1/count2/exposure2, or variables.outcome + group (+ exposure)")
    notes = [f"{n_drop} row(s) with a missing count, exposure or group excluded"] if n_drop else []
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
        warnings=notes,
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


# ---------------------------------------------------------------------------
# Zero-inflated count model (judgment: count distribution, zero-process formula)
# ---------------------------------------------------------------------------
@procedure("zip", requires=("outcome|formula",), decisions=("count_distribution", "inflation_formula"))
def zip_model(df: pd.DataFrame, spec: Spec) -> TestResult:
    import patsy
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson

    f = _formula(spec, df)
    yname = spec.variables.outcome or f.split("~")[0].strip()
    dist = str(spec.decisions["count_distribution"]).lower()
    if dist not in ("poisson", "negbin", "negative_binomial", "nb"):
        raise SpecError("decisions.count_distribution must be 'poisson' or 'negbin'")
    dist = "poisson" if dist == "poisson" else "negbin"
    infl = str(spec.decisions["inflation_formula"]).strip() or "1"
    d = df.dropna(subset=[yname]).copy()
    y, X = patsy.dmatrices(f, d, return_type="dataframe")
    Xi = patsy.dmatrix(infl, d.loc[X.index], return_type="dataframe")
    y = y.iloc[:, 0].to_numpy(int)
    ex = spec.variables.exposure
    kw = {"exposure": d.loc[X.index, ex].to_numpy(float)} if ex else {}
    zp = ZeroInflatedPoisson(y, X, exog_infl=Xi, inflation="logit", **kw).fit(disp=0, maxiter=1000, method="bfgs")
    warnings = []
    if dist == "negbin":
        m0 = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Xi, inflation="logit", **kw)
        res = m0.fit(start_params=np.r_[zp.params.to_numpy(), 0.5], disp=0, maxiter=2000, method="nm")
        try:
            res2 = m0.fit(start_params=res.params, disp=0, maxiter=2000, method="bfgs")
            if np.isfinite(res2.bse).all() and np.isfinite(res2.llf):
                res = res2
            else:
                warnings.append("ZINB: BFGS refinement produced non-finite standard errors (alpha near the boundary); Nelder–Mead fit kept.")
        except Exception as e:  # pragma: no cover
            warnings.append(f"ZINB refinement failed: {e}")
        label = "zero-inflated negative binomial"
    else:
        res, label = zp, "zero-inflated Poisson"
    nb = sm.NegativeBinomial(y, X, **kw).fit(disp=0, maxiter=500)
    pois = sm.Poisson(y, X, **kw).fit(disp=0, maxiter=500)
    delta = float(nb.aic - res.aic)
    # cross-check: the count-part coefficients should point the same way as a plain NB fit; the AIC
    # comparison the taxonomy asks for is reported in the beats_nb check, not as an agreement rule
    count_names = [nm for nm in res.params.index if not nm.startswith("inflate_") and nm != "alpha" and nm != "const"]
    cc = compare(
        "plain negative binomial (count-part coefficient directions; AIC in the beats_nb check)",
        "If the zero-inflated model doesn't clearly beat plain negative binomial, the extra zeros may just be overdispersion.",
        {f"direction[{nm}]": float(res.params[nm]) for nm in count_names},
        {f"direction[{nm}]": float(nb.params[nm]) for nm in count_names},
        kind="direction",
        detail=f"ΔAIC (NB − ZI) = {delta:.2f}",
    )
    obs0 = float((y == 0).mean())
    pred0_nb = float(np.mean(nb.predict(which="prob", y_values=[0]))) if hasattr(nb, "predict") else float("nan")
    try:
        pred0_nb = float(np.mean(nb.model.predict(nb.params, which="prob")[:, 0]))
    except Exception:
        pass
    chk = [
        checks.manual("zero_process", "A real zero-generating process exists. That's a design question, not a statistical one", f"observed zero fraction {obs0:.3f}" + (f"; NB predicts {pred0_nb:.3f}" if np.isfinite(pred0_nb) else "")),
        CheckResult("beats_nb", "Clearly better than plain negative binomial (ΔAIC > 2)", bool(delta > 2), delta, None, f"AIC: zero-inflated {float(res.aic):.1f}, NB {float(nb.aic):.1f}, Poisson {float(pois.aic):.1f}"),
    ]
    if chk[1].passed is False:
        warnings.append("Zero inflation adds little over negative binomial; prefer the simpler model, or a hurdle model if zeros are a separate process.")
    names = list(res.params.index)
    ci = res.conf_int(alpha=1 - spec.ci_level)
    ests = []
    for i, nm in enumerate(names):
        b, lo, hi = float(res.params.iloc[i]), float(ci.iloc[i, 0]), float(ci.iloc[i, 1])
        p = float(res.pvalues.iloc[i])
        if nm.startswith("inflate_"):
            ests.append(Estimate(f"zero part OR {nm[8:]}", float(np.exp(b)), (float(np.exp(lo)), float(np.exp(hi))), spec.ci_level, f"p = {p:.3g}"))
        elif nm == "alpha":
            ests.append(Estimate("NB dispersion α", b, (lo, hi), spec.ci_level))
        else:
            ests.append(Estimate(f"count part RR {nm}", float(np.exp(b)), (float(np.exp(lo)), float(np.exp(hi))), spec.ci_level, f"p = {p:.3g}"))
    return TestResult(
        test="zip",
        name=f"{label} model",
        n=int(len(y)),
        statistic_name="",
        statistic=None,
        p=None,
        estimates=ests,
        checks=chk,
        cross_check=cc,
        warnings=warnings,
        extra={"formula": f, "inflation_formula": infl, "aic": {"zero_inflated": float(res.aic), "negbin": float(nb.aic), "poisson": float(pois.aic)}, "converged": bool(res.mle_retvals.get("converged", True))},
    )
