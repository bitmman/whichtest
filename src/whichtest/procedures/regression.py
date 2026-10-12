"""Regression models: simple and multiple linear, logistic, multinomial logistic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .. import checks
from ..data import column
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


def _formula(spec: Spec, df: pd.DataFrame) -> str:
    v = spec.variables
    if v.formula:
        return v.formula
    if not v.outcome or not v.predictors:
        raise SpecError(f"{spec.test} needs variables.formula, or variables.outcome + variables.predictors")
    cat = set(v.categorical)
    for c in [v.outcome, *v.predictors]:
        if c not in df.columns:
            raise SpecError(f"column {c!r} is not in the data")
    for p in v.predictors:
        if p not in cat and not pd.api.types.is_numeric_dtype(df[p]):
            cat.add(p)
    rhs = " + ".join(f"C({p})" if p in cat else p for p in v.predictors)
    return f"{v.outcome} ~ {rhs}"


def _coef_estimates(m, level: float, exp: bool = False, name: str = "") -> list[Estimate]:
    ci = m.conf_int(alpha=1 - level)
    out = []
    for term in m.params.index:
        b, lo, hi = float(m.params[term]), float(ci.loc[term, 0]), float(ci.loc[term, 1])
        if exp:
            b, lo, hi = np.exp(b), np.exp(lo), np.exp(hi)
        out.append(Estimate(f"{name}{term}", b, (lo, hi), level, f"p = {float(m.pvalues[term]):.3g}"))
    return out


@procedure("simple_lin_reg", requires=("x|predictors|formula", "y|outcome|formula"))
def simple_lin_reg(df: pd.DataFrame, spec: Spec) -> TestResult:
    v = spec.variables
    xname = v.x or (v.predictors[0] if v.predictors else None)
    yname = v.y or v.outcome
    if v.formula:
        m = smf.ols(v.formula, df).fit()
        if m.model.exog.shape[1] != 2:
            raise SpecError("simple_lin_reg needs exactly one predictor; use multiple_reg")
        yname, xname = m.model.endog_names, m.model.exog_names[1]
        y, x = m.model.endog, m.model.exog[:, 1]
    else:
        x = column(df, xname, "x")
        y = column(df, yname, "y")
        keep = ~(np.isnan(x) | np.isnan(y))
        x, y = x[keep], y[keep]
        m = sm.OLS(y, sm.add_constant(x)).fit()
    res = stats.linregress(x, y, alternative=spec.alternative)
    cc = compare(
        "statsmodels OLS",
        "Slope, intercept and p-value should match linregress exactly.",
        {"slope": res.slope, "intercept": res.intercept, "p": res.pvalue, "r2": res.rvalue**2},
        {"slope": m.params[1], "intercept": m.params[0], "p": m.pvalues[1], "r2": m.rsquared},
        rtol=1e-6,
    )
    ci = np.asarray(m.conf_int(alpha=1 - spec.ci_level))
    cooks = m.get_influence().cooks_distance[0]
    thr = 4 / len(y)
    n_infl = int((cooks > thr).sum())
    chk = [
        checks.manual("linearity", "Relationship is linear (inspect a scatter plot)", "residuals vs fitted should show no curve"),
        checks.shapiro(np.asarray(m.resid), "residual_normality", "Residuals roughly normal", spec.alpha),
        checks.breusch_pagan(np.asarray(m.resid), m.model.exog, "homoscedasticity", "Constant residual variance", spec.alpha),
        CheckResult("influence", "No single point drives the fit", bool(n_infl == 0), float(cooks.max()), None, f"{n_infl} point(s) with Cook's D > 4/n = {thr:.3f}; max D = {cooks.max():.3f}"),
    ]
    warnings = []
    if chk[2].passed is False:
        hc3 = m.get_robustcov_results(cov_type="HC3")
        warnings.append(f"Heteroscedastic: with HC3 standard errors the slope p = {float(hc3.pvalues[1]):.4g}.")
    return TestResult(
        test="simple_lin_reg",
        n=int(len(y)),
        statistic_name="t (slope)",
        statistic=float(m.tvalues[1]),
        p=float(res.pvalue),
        df=float(m.df_resid),
        estimates=[
            Estimate(f"slope ({yname} on {xname})", float(res.slope), (float(ci[1, 0]), float(ci[1, 1])), spec.ci_level),
            Estimate("intercept", float(res.intercept), (float(ci[0, 0]), float(ci[0, 1])), spec.ci_level),
        ],
        effect_size=Estimate("R²", float(m.rsquared)),
        checks=chk,
        cross_check=cc,
        warnings=warnings,
    )


@procedure("multiple_reg", requires=("outcome|formula",))
def multiple_reg(df: pd.DataFrame, spec: Spec) -> TestResult:
    f = _formula(spec, df)
    m = smf.ols(f, df).fit()
    beta = np.linalg.lstsq(m.model.exog, m.model.endog, rcond=None)[0]
    cc = compare(
        "numpy.linalg.lstsq on the design matrix",
        "A raw least-squares solve should reproduce m.params.",
        {n: float(b) for n, b in zip(m.model.exog_names, m.params)},
        {n: float(b) for n, b in zip(m.model.exog_names, beta)},
        rtol=1e-6,
        atol=1e-8,
    )
    chk = [
        checks.shapiro(np.asarray(m.resid), "residual_normality", "Residuals roughly normal", spec.alpha),
        checks.breusch_pagan(np.asarray(m.resid), m.model.exog, "homoscedasticity", "Constant residual variance", spec.alpha),
        checks.vif(m.model.exog, m.model.exog_names, "multicollinearity", "Predictors not collinear (VIF < 5)"),
    ]
    # homogeneous slopes (ANCOVA): group × covariate interactions should be non-significant
    v = spec.variables
    cat = [p for p in v.predictors if p in v.categorical or not pd.api.types.is_numeric_dtype(df[p])]
    num = [p for p in v.predictors if p not in cat]
    if cat and num and not v.formula:
        rhs = " + ".join(f"C({p})" for p in cat) + " + " + " + ".join(num)
        inter = " + ".join(f"C({g}):{x}" for g in cat for x in num)
        m_int = smf.ols(f"{v.outcome} ~ {rhs} + {inter}", df).fit()
        from statsmodels.stats.anova import anova_lm

        cmp = anova_lm(m, m_int)
        p_int = float(cmp["Pr(>F)"].iloc[1])
        chk.append(CheckResult("homogeneous_slopes", "Covariate slopes equal across groups (no group × covariate interaction)", bool(p_int >= spec.alpha), float(cmp["F"].iloc[1]), p_int, "F-test of the interaction model against the additive model"))
    else:
        chk.append(checks.manual("homogeneous_slopes", "Covariate slopes equal across groups", "only testable with a categorical and a continuous predictor"))
    warnings = []
    extra = {"formula": f, "adj_r2": float(m.rsquared_adj), "F": float(m.fvalue), "F_p": float(m.f_pvalue)}
    if chk[1].passed is False:
        hc3 = m.get_robustcov_results(cov_type="HC3")
        extra["hc3_pvalues"] = {n: float(p) for n, p in zip(m.model.exog_names, hc3.pvalues)}
        warnings.append("Heteroscedastic: HC3-robust p-values are in extra.hc3_pvalues.")
    # partial eta² for each term from a type-II ANOVA table
    try:
        tab = sm.stats.anova_lm(m, typ=2)
        ss_res = float(tab.loc["Residual", "sum_sq"])
        extra["partial_eta_sq"] = {str(t): float(tab.loc[t, "sum_sq"] / (tab.loc[t, "sum_sq"] + ss_res)) for t in tab.index if t != "Residual"}
    except Exception:  # pragma: no cover
        pass
    return TestResult(
        test="multiple_reg",
        n=int(m.nobs),
        statistic_name="F (model)",
        statistic=float(m.fvalue),
        p=float(m.f_pvalue),
        df=(float(m.df_model), float(m.df_resid)),
        estimates=_coef_estimates(m, spec.ci_level),
        effect_size=Estimate("adjusted R²", float(m.rsquared_adj)),
        checks=chk,
        cross_check=cc,
        warnings=warnings,
        extra=extra,
    )


@procedure("logistic", requires=("outcome|formula",))
def logistic(df: pd.DataFrame, spec: Spec) -> TestResult:
    f = _formula(spec, df)
    m = smf.logit(f, df).fit(disp=0, maxiter=200)
    g = smf.glm(f, df, family=sm.families.Binomial()).fit()
    cc = compare(
        "statsmodels GLM Binomial",
        "Logit and binomial GLM are the same model fitted two ways. Coefficients should match to many decimals.",
        {n: float(b) for n, b in zip(m.params.index, m.params)},
        {n: float(b) for n, b in zip(g.params.index, g.params)},
        rtol=1e-5,
        atol=1e-6,
    )
    y = np.asarray(m.model.endog)
    events = int(min(y.sum(), len(y) - y.sum()))
    k = int(m.model.exog.shape[1] - 1)
    epv = events / k if k else float("inf")
    fitted = np.asarray(m.predict())
    sep = bool((np.abs(m.params) > 10).any() or np.isclose(fitted, 0, atol=1e-8).any() or np.isclose(fitted, 1, atol=1e-8).any())
    chk = [
        CheckResult("events_per_variable", "≥ 10 events per predictor (minority class / predictors)", bool(epv >= 10), float(epv), None, f"{events} events in the smaller class, {k} predictor terms → EPV = {epv:.1f}"),
        CheckResult("separation", "No perfect or quasi-perfect separation", not sep, None, None, "coefficients or fitted probabilities at the boundary suggest separation" if sep else "no sign of separation"),
        checks.manual("logit_linearity", "Continuous predictors linear on the logit scale", "check with splines or a Box–Tidwell term"),
        checks.vif(m.model.exog, m.model.exog_names, "multicollinearity", "Predictors not collinear (VIF < 5)"),
    ]
    warnings = []
    if sep or epv < 10:
        warnings.append("Consider Firth's penalized logistic regression (firthlogist) for separation or rare events.")
    return TestResult(
        test="logistic",
        n=int(m.nobs),
        statistic_name="LR χ²",
        statistic=float(m.llr),
        p=float(m.llr_pvalue),
        df=float(m.df_model),
        estimates=_coef_estimates(m, spec.ci_level, exp=True, name="OR "),
        effect_size=Estimate("McFadden pseudo-R²", float(m.prsquared)),
        checks=chk,
        cross_check=cc,
        warnings=warnings,
        extra={"formula": f, "coef": {n: float(b) for n, b in zip(m.params.index, m.params)}, "aic": float(m.aic)},
    )


@procedure("mnlogit", requires=("outcome|formula",))
def mnlogit(df: pd.DataFrame, spec: Spec) -> TestResult:
    f = _formula(spec, df)
    d = df.copy()
    yname = spec.variables.outcome or f.split("~")[0].strip()
    cats = pd.Categorical(d[yname])
    d[yname] = cats.codes
    m = smf.mnlogit(f, d).fit(disp=0, maxiter=500)
    probs = np.asarray(m.predict())
    cc = None
    try:
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(m.model.exog)[:, 1:]  # sklearn adds its own intercept
        y = np.asarray(m.model.endog).astype(int)
        lr = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(X, y)
        p2 = lr.predict_proba(X)
        maxdiff = float(np.abs(probs - p2).max())
        ll1 = float(m.llf)
        ll2 = float(np.log(p2[np.arange(len(y)), y]).sum())
        cc = compare(
            "sklearn LogisticRegression (unpenalized) predicted probabilities",
            "The unpenalized fit has the same likelihood but a different parameterization. Compare predicted probabilities, not coefficients.",
            {"log_likelihood": ll1, "max_prob_diff": 0.0},
            {"log_likelihood": ll2, "max_prob_diff": maxdiff},
            rtol=1e-4,
            atol=1e-3,
        )
    except ImportError:
        pass
    counts = pd.Series(np.asarray(m.model.endog).astype(int)).value_counts().sort_index()
    k = int(m.model.exog.shape[1] - 1)
    epc = float(counts.min() / k) if k else float("inf")
    chk = [
        checks.manual("iia", "Independence of irrelevant alternatives (odds between two categories unaffected by the others)"),
        CheckResult("events_per_category", "≥ 10 cases per predictor in the smallest outcome category", bool(epc >= 10), epc, None, f"smallest category n = {int(counts.min())}, {k} predictor terms"),
    ]
    ests = []
    z = stats.norm.ppf(0.5 + spec.ci_level / 2)
    ref = str(cats.categories[0])
    for j, col in enumerate(m.params.columns):
        label = str(cats.categories[j + 1])
        for term in m.params.index:
            b, se = float(m.params.loc[term, col]), float(m.bse.loc[term, col])
            lo, hi = b - z * se, b + z * se
            ests.append(Estimate(f"RRR[{label} vs {ref}] {term}", float(np.exp(b)), (float(np.exp(lo)), float(np.exp(hi))), spec.ci_level, f"p = {float(m.pvalues.loc[term, col]):.3g}"))
    return TestResult(
        test="mnlogit",
        n=int(m.nobs),
        statistic_name="LR χ²",
        statistic=float(m.llr),
        p=float(m.llr_pvalue),
        df=float(m.df_model),
        estimates=ests,
        effect_size=Estimate("McFadden pseudo-R²", float(m.prsquared)),
        checks=chk,
        cross_check=cc,
        extra={"formula": f, "categories": [str(c) for c in cats.categories], "reference": ref},
    )
