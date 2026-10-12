"""Correlations: Pearson, Spearman, point-biserial."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import checks, effects
from ..data import column
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


def _xy(df: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray, str, str]:
    v = spec.variables
    xn, yn = v.x or (v.columns[0] if v.columns else None), v.y or (v.columns[1] if v.columns and len(v.columns) > 1 else v.outcome)
    x, y = column(df, xn, "x"), column(df, yn, "y")
    keep = ~(np.isnan(x) | np.isnan(y))
    return x[keep], y[keep], xn, yn


@procedure("pearson", requires=("x|columns", "y|columns|outcome"))
def pearson(df: pd.DataFrame, spec: Spec) -> TestResult:
    x, y, xn, yn = _xy(df, spec)
    res = stats.pearsonr(x, y, alternative=spec.alternative)
    ci = res.confidence_interval(spec.ci_level)
    r2 = float(np.corrcoef(x, y)[0, 1])
    cc = compare("numpy.corrcoef", "The coefficient must match exactly.", {"r": res.statistic}, {"r": r2})
    mz = np.abs(stats.zscore(np.column_stack([x, y]), ddof=1)).max(axis=1)
    n_out = int((mz > 3).sum())
    chk = [
        checks.manual("linearity", "Relationship is linear (inspect a scatter plot)"),
        CheckResult("outliers", "No extreme points driving r", bool(n_out == 0), float(mz.max()), None, f"{n_out} point(s) with |z| > 3 on either variable"),
        checks.shapiro_groups({xn: x, yn: y}, "bivariate_normality", "Each variable roughly normal (needed for the p-value and CI)", spec.alpha),
    ]
    return TestResult(
        test="pearson",
        n=int(len(x)),
        statistic_name="r",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(len(x) - 2),
        effect_size=Estimate("r", float(res.statistic), (float(ci.low), float(ci.high)), spec.ci_level, "Fisher z"),
        estimates=[Estimate("R²", float(res.statistic) ** 2)],
        checks=chk,
        cross_check=cc,
    )


@procedure("spearman", requires=("x|columns", "y|columns|outcome"))
def spearman(df: pd.DataFrame, spec: Spec) -> TestResult:
    x, y, xn, yn = _xy(df, spec)
    res = stats.spearmanr(x, y, alternative=spec.alternative)
    r2 = stats.pearsonr(stats.rankdata(x), stats.rankdata(y), alternative=spec.alternative)
    cc = compare(
        "scipy.stats.pearsonr on ranks",
        "Spearman's ρ is Pearson's r on ranks. The coefficients must match exactly.",
        {"rho": res.statistic, "p": res.pvalue},
        {"rho": r2.statistic, "p": r2.pvalue},
        rtol=1e-6,
        atol=1e-8,
    )
    rho = float(res.statistic)
    ci = effects.fisher_z_ci(rho, len(x), spec.ci_level, "ρ")
    ties = float(max(1 - len(np.unique(x)) / len(x), 1 - len(np.unique(y)) / len(y)))
    tau = stats.kendalltau(x, y, alternative=spec.alternative)
    warnings = []
    if len(x) < 20 or ties > 0.2:
        warnings.append(f"Small sample or many ties ({ties:.0%}): Kendall's τ = {tau.statistic:.3f}, p = {tau.pvalue:.4g}.")
    # monotonicity: compare Spearman with the correlation of the LOWESS-like binned means (crude)
    return TestResult(
        test="spearman",
        n=int(len(x)),
        statistic_name="ρ",
        statistic=rho,
        p=float(res.pvalue),
        effect_size=Estimate("ρ", rho, ci.ci, spec.ci_level, "Fisher z (approximate)"),
        estimates=[Estimate("Kendall's τ", float(tau.statistic), note=f"p = {tau.pvalue:.3g}")],
        checks=[checks.manual("monotonic", "Relationship is monotonic (inspect a scatter plot)")],
        cross_check=cc,
        warnings=warnings,
    )


@procedure("point_biserial", requires=("outcome|x", "group|y"))
def point_biserial(df: pd.DataFrame, spec: Spec) -> TestResult:
    v = spec.variables
    cont = column(df, v.outcome or v.x, "outcome")
    bname = v.group or v.y
    b = column(df, bname, "group", numeric=False)
    keep = ~np.isnan(cont) & ~pd.isna(b)
    cont, b = cont[keep], b[keep]
    levels = list(pd.unique(b))
    if len(levels) != 2:
        raise SpecError(f"point_biserial needs a binary variable; {bname!r} has {len(levels)} levels")
    bin_ = (b == levels[1]).astype(float)  # 1 = second level
    res = stats.pointbiserialr(bin_, cont)
    t = stats.ttest_ind(cont[bin_ == 1], cont[bin_ == 0], equal_var=True)
    cc = compare(
        "scipy.stats.ttest_ind(equal_var=True)",
        "Mathematically identical to Student's t-test. Same p-value.",
        {"p": res.pvalue},
        {"p": t.pvalue},
    )
    g = {str(levels[0]): cont[bin_ == 0], str(levels[1]): cont[bin_ == 1]}
    return TestResult(
        test="point_biserial",
        n=int(len(cont)),
        statistic_name="r_pb",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        effect_size=effects.fisher_z_ci(float(res.statistic), len(cont), spec.ci_level, "r_pb"),
        estimates=[Estimate("Student t", float(t.statistic), note=f"df = {len(cont) - 2}; r_pb > 0 means {levels[1]!s} has the higher mean")],
        checks=[
            checks.shapiro_groups(g, "normality_by_group", "Continuous variable roughly normal within each level", spec.alpha),
            checks.levene(list(g.values()), "equal_variance", "Equal variance across the two levels (Student's t assumption)", spec.alpha),
        ],
        cross_check=cc,
    )
