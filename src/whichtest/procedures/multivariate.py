"""Several outcomes at once: MANOVA, MANCOVA, multivariate multiple regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA

from .. import checks
from ..registry import procedure
from ..result import CheckResult, CrossCheck, Comparison, Estimate, TestResult, compare
from ..spec import Spec, SpecError


def _outcomes(spec: Spec, df: pd.DataFrame) -> list[str]:
    cols = spec.variables.columns or []
    if len(cols) < 2:
        raise SpecError(f"{spec.test} needs variables.columns listing two or more outcome columns")
    for c in cols:
        if c not in df.columns:
            raise SpecError(f"column {c!r} is not in the data")
    return cols


def _rhs(spec: Spec, df: pd.DataFrame, extra: list[str] = ()) -> tuple[str, list[str]]:
    v = spec.variables
    terms: list[str] = []
    if v.group:
        terms.append(f"C({v.group})")
    cat = set(v.categorical)
    for p in [*v.predictors, *extra]:
        if p not in df.columns:
            raise SpecError(f"column {p!r} is not in the data")
        terms.append(f"C({p})" if (p in cat or not pd.api.types.is_numeric_dtype(df[p])) else p)
    if not terms:
        raise SpecError(f"{spec.test} needs variables.group and/or variables.predictors")
    return " + ".join(terms), terms


def _mv_stats(res, term: str) -> pd.DataFrame:
    return res.results[term]["stat"]


def _pillai(tab: pd.DataFrame) -> tuple[float, float, float, float, float]:
    r = tab.loc["Pillai's trace"]
    return float(r["Value"]), float(r["F Value"]), float(r["Num DF"]), float(r["Den DF"]), float(r["Pr > F"])


def _residual_checks(fits: dict[str, object], df: pd.DataFrame, group: str | None, outcomes: list[str], alpha: float, with_box: bool = True) -> list[CheckResult]:
    resid = np.column_stack([np.asarray(f.resid) for f in fits.values()])
    out: list[CheckResult] = []
    try:
        import pingouin as pg

        hz = pg.multivariate_normality(pd.DataFrame(resid, columns=outcomes), alpha=alpha)
        out.append(CheckResult("multivariate_normality", "Residuals roughly multivariate normal", bool(hz.normal), float(hz.hz), float(hz.pval), "Henze–Zirkler test on the residuals"))
    except ImportError:
        out.append(checks.manual("multivariate_normality", "Residuals roughly multivariate normal", "pingouin not installed"))
    if with_box:
        if group is None:
            out.append(checks.manual("covariance_homogeneity", "Similar covariance matrices across groups", "no grouping variable"))
        else:
            try:
                import pingouin as pg

                bm = pg.box_m(df[[*outcomes, group]].dropna(), dvs=outcomes, group=group)
                out.append(CheckResult("covariance_homogeneity", "Similar covariance matrices across groups (Box's M)", bool(bm["equal_cov"].iloc[0]), float(bm["Chi2"].iloc[0]), float(bm["pval"].iloc[0]), f"Box's M χ² on {int(bm['df'].iloc[0])} df"))
            except ImportError:
                out.append(checks.manual("covariance_homogeneity", "Similar covariance matrices across groups", "pingouin not installed"))
    return out


def _univariate_posthoc(fits: dict[str, object], term: str, alpha: float) -> dict:
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multitest import multipletests

    rows = []
    for y, f in fits.items():
        tab = anova_lm(f, typ=2)
        rows.append({"outcome": y, "F": float(tab.loc[term, "F"]), "p": float(tab.loc[term, "PR(>F)"])})
    adj = multipletests([r["p"] for r in rows], method="holm")[1]
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return {"method": f"univariate ANOVA per outcome for {term}, Holm-adjusted", "outcomes": rows}


@procedure("manova", requires=("columns", "group"))
def manova(df: pd.DataFrame, spec: Spec) -> TestResult:
    ys = _outcomes(spec, df)
    g = spec.variables.group
    rhs, terms = _rhs(spec, df)
    term = f"C({g})"
    d = df[[*ys, g, *spec.variables.predictors]].dropna()
    res = MANOVA.from_formula(f"{' + '.join(ys)} ~ {rhs}", d).mv_test()
    tab = _mv_stats(res, term)
    V, F, df1, df2, p = _pillai(tab)
    wilks = tab.loc["Wilks' lambda"]
    cc = compare(
        "Wilks' λ (and the other two statistics) from the same mv_test",
        "All four statistics usually agree. If Pillai's trace and Wilks' λ disagree, assumptions are likely violated; trust Pillai.",
        {"p": p, "p_hotelling": p, "p_roy": p},
        {"p": float(wilks["Pr > F"]), "p_hotelling": float(tab.loc["Hotelling-Lawley trace", "Pr > F"]), "p_roy": float(tab.loc["Roy's greatest root", "Pr > F"])},
        kind="approximate",
        alpha=spec.alpha,
    )
    fits = {y: smf.ols(f"{y} ~ {rhs}", d).fit() for y in ys}
    chk = _residual_checks(fits, d, g, ys, spec.alpha)
    corr = d[ys].corr().to_numpy()
    off = np.abs(corr[np.triu_indices(len(ys), 1)])
    chk.append(CheckResult("outcome_redundancy", "Outcomes correlated, but not redundant", bool(off.max() < 0.9), float(off.max()), None, f"max |r| between outcomes = {off.max():.2f}" + ("" if off.min() > 0.1 else "; some outcomes barely correlate, a joint test adds little")))
    k = d[g].nunique()
    s = min(len(ys), k - 1)
    eta_p = V / s if s > 0 else None
    return TestResult(
        test="manova",
        n={str(l): int(c) for l, c in d[g].value_counts().items()},
        statistic_name="Pillai's trace F",
        statistic=F,
        p=p,
        df=(df1, df2),
        estimates=[Estimate("Pillai's trace", V), Estimate("Wilks' λ", float(wilks["Value"]))],
        effect_size=Estimate("partial η² (Pillai / s)", eta_p),
        checks=chk,
        cross_check=cc,
        post_hoc=_univariate_posthoc(fits, term, spec.alpha),
        extra={"mv_table": {str(i): {c: float(v) for c, v in r.items()} for i, r in tab.iterrows()}},
    )


@procedure("mancova", requires=("columns", "group", "predictors"))
def mancova(df: pd.DataFrame, spec: Spec) -> TestResult:
    ys = _outcomes(spec, df)
    g = spec.variables.group
    rhs, terms = _rhs(spec, df)
    term = f"C({g})"
    d = df[[*ys, g, *spec.variables.predictors]].dropna()
    res = MANOVA.from_formula(f"{' + '.join(ys)} ~ {rhs}", d).mv_test()
    tab = _mv_stats(res, term)
    V, F, df1, df2, p = _pillai(tab)
    fits = {y: smf.ols(f"{y} ~ {rhs}", d).fit() for y in ys}
    post = _univariate_posthoc(fits, term, spec.alpha)
    uni_p = [r["p_holm"] for r in post["outcomes"]]
    joint_sig = p < spec.alpha
    any_uni = any(q < spec.alpha for q in uni_p)
    agrees = not (joint_sig and not any_uni)
    cc = CrossCheck(
        method="per-outcome ANCOVA (OLS with the same right-hand side)",
        why="Univariate ANCOVAs don't replace the joint test, but a joint effect with no univariate signal at all deserves a closer look.",
        kind="direction",
        agrees=agrees,
        comparisons=[Comparison("p", p, float(min(uni_p)), agrees, "joint significant ⇒ some univariate (Holm) significant")],
        detail=f"smallest Holm-adjusted univariate p = {min(uni_p):.4g}",
    )
    chk = _residual_checks(fits, d, g, ys, spec.alpha)
    # homogeneous slopes: group × covariate interaction in the joint model
    covs = [t for t in terms if not t.startswith("C(")]
    if covs:
        inter = " + ".join(f"{term}:{c}" for c in covs)
        res_i = MANOVA.from_formula(f"{' + '.join(ys)} ~ {rhs} + {inter}", d).mv_test()
        ps = [float(_mv_stats(res_i, f"{term}:{c}").loc["Pillai's trace", "Pr > F"]) for c in covs]
        chk.append(CheckResult("homogeneous_slopes", "No group × covariate interaction", bool(min(ps) >= spec.alpha), None, float(min(ps)), "Pillai test of each group × covariate interaction; smallest p shown"))
    else:
        chk.append(checks.manual("homogeneous_slopes", "No group × covariate interaction", "no continuous covariate"))
    k = d[g].nunique()
    s = min(len(ys), k - 1)
    # adjusted group differences per outcome
    ests = [Estimate("Pillai's trace", V)]
    for y, f in fits.items():
        ci = f.conf_int(alpha=1 - spec.ci_level)
        for name in f.params.index:
            if name.startswith(term):
                ests.append(Estimate(f"adjusted difference {y}: {name}", float(f.params[name]), (float(ci.loc[name, 0]), float(ci.loc[name, 1])), spec.ci_level, f"p = {float(f.pvalues[name]):.3g}"))
    return TestResult(
        test="mancova",
        n={str(l): int(c) for l, c in d[g].value_counts().items()},
        statistic_name="Pillai's trace F",
        statistic=F,
        p=p,
        df=(df1, df2),
        estimates=ests,
        effect_size=Estimate("partial η² (Pillai / s)", V / s if s > 0 else None),
        checks=chk,
        cross_check=cc,
        post_hoc=post,
        extra={"mv_table": {str(i): {c: float(v) for c, v in r.items()} for i, r in tab.iterrows()}},
    )


@procedure("mv_reg", requires=("columns", "predictors"))
def mv_reg(df: pd.DataFrame, spec: Spec) -> TestResult:
    ys = _outcomes(spec, df)
    rhs, terms = _rhs(spec, df)
    d = df[[*ys, *spec.variables.predictors, *([spec.variables.group] if spec.variables.group else [])]].dropna()
    res = MANOVA.from_formula(f"{' + '.join(ys)} ~ {rhs}", d).mv_test()
    fits = {y: smf.ols(f"{y} ~ {rhs}", d).fit() for y in ys}
    X = next(iter(fits.values())).model.exog
    Y = np.column_stack([np.asarray(f.model.endog) for f in fits.values()])
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    names = next(iter(fits.values())).model.exog_names
    cc = compare(
        "numpy.linalg.lstsq(X, Y)",
        "Coefficients are identical to the separate OLS fits. Only the joint tests differ.",
        {f"{y}:{nm}": float(f.params[nm]) for y, f in fits.items() for nm in names},
        {f"{y}:{nm}": float(B[i, j]) for j, y in enumerate(ys) for i, nm in enumerate(names)},
        rtol=1e-6,
        atol=1e-8,
    )
    chk = _residual_checks(fits, d, None, ys, spec.alpha, with_box=False)
    chk.append(checks.vif(X, names, "multicollinearity", "Predictors not too collinear (VIF < 5)"))
    ests = []
    joint = {}
    for t in terms:
        tab = _mv_stats(res, t)
        V, F, df1, df2, p = _pillai(tab)
        joint[t] = {"pillai": V, "F": F, "df": (df1, df2), "p": p}
        ests.append(Estimate(f"joint test {t}: Pillai F", F, note=f"df = ({df1:g}, {df2:g}), p = {p:.3g}"))
    for y, f in fits.items():
        ci = f.conf_int(alpha=1 - spec.ci_level)
        for nm in f.params.index:
            if nm == "Intercept":
                continue
            ests.append(Estimate(f"{y} ~ {nm}", float(f.params[nm]), (float(ci.loc[nm, 0]), float(ci.loc[nm, 1])), spec.ci_level, f"p = {float(f.pvalues[nm]):.3g}"))
    first = terms[0]
    return TestResult(
        test="mv_reg",
        n=int(len(d)),
        statistic_name=f"Pillai F ({first})",
        statistic=joint[first]["F"],
        p=joint[first]["p"],
        df=joint[first]["df"],
        estimates=ests,
        effect_size=Estimate("R² per outcome", None, note="; ".join(f"{y}: {f.rsquared:.3f}" for y, f in fits.items())),
        checks=chk,
        cross_check=cc,
        extra={"joint_tests": joint},
    )
