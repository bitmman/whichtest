"""Parametric comparisons of means: one-sample, Welch, paired, one-way ANOVA."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .. import checks, effects
from ..data import column, groups, paired
from ..registry import procedure
from ..result import Estimate, TestResult, compare
from ..spec import Spec


@procedure("one_sample_t", requires=("outcome",))
def one_sample_t(df: pd.DataFrame, spec: Spec) -> TestResult:
    x = column(df, spec.variables.outcome, "outcome")
    x = x[~np.isnan(x)]
    mu0 = float(spec.params.get("mu0", 0.0))
    res = stats.ttest_1samp(x, popmean=mu0, alternative=spec.alternative)
    ci = res.confidence_interval(spec.ci_level)

    # cross-check: statsmodels DescrStatsW
    alt = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}[spec.alternative]
    t2, p2, df2 = sm.stats.DescrStatsW(x).ttest_mean(mu0, alternative=alt)
    cc = compare(
        "statsmodels.stats.weightstats.DescrStatsW.ttest_mean",
        "Same test in a second library. The t statistic and p-value should match to many decimals.",
        {"t": res.statistic, "p": res.pvalue, "df": res.df},
        {"t": t2, "p": p2, "df": df2},
    )
    return TestResult(
        test="one_sample_t",
        n=int(len(x)),
        statistic_name="t",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(res.df),
        estimates=[Estimate("mean − μ₀", float(np.mean(x) - mu0), (float(ci.low - mu0), float(ci.high - mu0)), spec.ci_level, f"μ₀ = {mu0:g}")],
        effect_size=effects.cohen_d_one(x, mu0),
        checks=[
            checks.shapiro(x, "normality", "Roughly normal, or n large enough for the CLT", spec.alpha),
            checks.outliers_z(x, "outliers", "No extreme values pulling the mean"),
        ],
        cross_check=cc,
    )


@procedure("indep_t", requires=("outcome", "group"))
def indep_t(df: pd.DataFrame, spec: Spec) -> TestResult:
    g = groups(df, spec.variables.outcome, spec.variables.group)
    if len(g) != 2:
        from ..spec import SpecError

        raise SpecError(f"indep_t needs exactly two groups; {spec.variables.group!r} has {len(g)}: {list(g)}")
    (la, a), (lb, b) = g.items()
    res = stats.ttest_ind(a, b, equal_var=False, alternative=spec.alternative)
    ci = res.confidence_interval(spec.ci_level)

    from statsmodels.stats.weightstats import ttest_ind as sm_ttest

    alt = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}[spec.alternative]
    t2, p2, df2 = sm_ttest(a, b, usevar="unequal", alternative=alt)
    cc = compare(
        "statsmodels.stats.weightstats.ttest_ind(usevar='unequal')",
        "Same Welch test in statsmodels. Statistic, degrees of freedom and p-value should match.",
        {"t": res.statistic, "p": res.pvalue, "df": res.df},
        {"t": t2, "p": p2, "df": df2},
    )
    return TestResult(
        test="indep_t",
        n={str(la): int(len(a)), str(lb): int(len(b))},
        statistic_name="Welch t",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(res.df),
        estimates=[Estimate(f"mean difference ({la} − {lb})", float(np.mean(a) - np.mean(b)), (float(ci.low), float(ci.high)), spec.ci_level)],
        effect_size=effects.hedges_g(a, b, spec.ci_level),
        checks=[
            checks.shapiro_groups(g, "normality_by_group", "Each group roughly normal", spec.alpha),
            checks.manual("independence", "One measurement per subject, no matching between groups"),
            checks.levene([a, b], "variance_info", "Unequal variances are fine: Welch doesn't assume them equal", spec.alpha, informational=True),
        ],
        cross_check=cc,
        warnings=["Welch's t is used regardless of the Levene result; it costs almost nothing when variances are equal."],
    )


@procedure("paired_t", requires=("before|columns|outcome",))
def paired_t(df: pd.DataFrame, spec: Spec) -> TestResult:
    a, b, (la, lb) = paired(df, spec)  # a = first, b = second; difference = second − first
    res = stats.ttest_rel(b, a, alternative=spec.alternative)
    ci = res.confidence_interval(spec.ci_level)
    d = b - a
    r2 = stats.ttest_1samp(d, 0, alternative=spec.alternative)
    cc = compare(
        "scipy.stats.ttest_1samp(after − before, 0)",
        "A paired t-test is a one-sample test on the differences. The result is identical.",
        {"t": res.statistic, "p": res.pvalue, "df": res.df},
        {"t": r2.statistic, "p": r2.pvalue, "df": r2.df},
    )
    return TestResult(
        test="paired_t",
        n=int(len(d)),
        statistic_name="t",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(res.df),
        estimates=[Estimate(f"mean difference ({lb} − {la})", float(np.mean(d)), (float(ci.low), float(ci.high)), spec.ci_level)],
        effect_size=effects.d_z(d),
        checks=[
            checks.shapiro(d, "difference_normality", "Paired differences roughly normal", spec.alpha),
            checks.manual("pair_alignment", "Rows are the same subject, measured twice"),
        ],
        cross_check=cc,
    )


@procedure("oneway_anova", requires=("outcome", "group"))
def oneway_anova(df: pd.DataFrame, spec: Spec) -> TestResult:
    import statsmodels.formula.api as smf
    from statsmodels.stats.oneway import anova_oneway

    g = groups(df, spec.variables.outcome, spec.variables.group)
    arrays = list(g.values())
    res = stats.f_oneway(*arrays)

    # cross-check: OLS with a categorical factor; the F must equal scipy's
    y = np.concatenate(arrays)
    lab = np.concatenate([[str(k)] * len(v) for k, v in g.items()])
    d = pd.DataFrame({"y": y, "group": lab})
    m = smf.ols("y ~ C(group)", d).fit()
    tab = sm.stats.anova_lm(m, typ=2)
    f2, p2 = float(tab.loc["C(group)", "F"]), float(tab.loc["C(group)", "PR(>F)"])
    cc = compare(
        "statsmodels OLS y ~ C(group) + anova_lm",
        "The regression framing gives the same F. It must equal scipy's.",
        {"F": res.statistic, "p": res.pvalue},
        {"F": f2, "p": p2},
    )

    ss_b, ss_w = float(tab.loc["C(group)", "sum_sq"]), float(tab.loc["Residual", "sum_sq"])
    df_b, df_w = float(tab.loc["C(group)", "df"]), float(tab.loc["Residual", "df"])
    eta, omega = effects.eta_omega(ss_b, ss_w, df_b, df_w)

    welch = anova_oneway(y, groups=lab, use_var="unequal", welch_correction=True)
    chk = [
        checks.shapiro(np.asarray(m.resid), "residual_normality", "Residuals roughly normal", spec.alpha),
        checks.levene(arrays, "equal_variance", "Similar variance across groups", spec.alpha),
    ]
    warnings = []
    if chk[1].passed is False:
        warnings.append(
            f"Variances differ: Welch's ANOVA gives F = {welch.statistic:.3f}, df = ({welch.df_num:.0f}, {welch.df_denom:.1f}), p = {welch.pvalue:.4g}."
        )

    # post hoc: Tukey HSD from statsmodels, cross-checked against scipy
    post = None
    if len(g) >= 3:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tk = pairwise_tukeyhsd(y, lab, alpha=spec.alpha)
        sc = stats.tukey_hsd(*arrays)
        labels = list(g)
        rows = []
        tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        for _, r in tk_df.iterrows():
            i, j = [str(k) for k in labels].index(str(r["group1"])), [str(k) for k in labels].index(str(r["group2"]))
            rows.append(
                {
                    "group1": r["group1"],
                    "group2": r["group2"],
                    "meandiff": float(r["meandiff"]),
                    "p_adj": float(r["p-adj"]),
                    "ci": (float(r["lower"]), float(r["upper"])),
                    "reject": bool(r["reject"]),
                    "scipy_p": float(sc.pvalue[i, j]),
                }
            )
        agree = all(abs(r["p_adj"] - r["scipy_p"]) < 2e-3 for r in rows)  # statsmodels rounds p-adj
        post = {"method": "Tukey HSD (statsmodels pairwise_tukeyhsd, cross-checked with scipy.stats.tukey_hsd)", "pairs": rows, "cross_check_agrees": agree}
        if not agree:
            warnings.append("Tukey HSD p-values from statsmodels and scipy disagree; inspect post_hoc.")

    return TestResult(
        test="oneway_anova",
        n={str(k): int(len(v)) for k, v in g.items()},
        statistic_name="F",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=(df_b, df_w),
        estimates=[Estimate(f"mean[{k}]", float(np.mean(v)), None) for k, v in g.items()],
        effect_size=omega,
        checks=chk,
        cross_check=cc,
        post_hoc=post,
        warnings=warnings,
        extra={"eta_sq": eta.value, "welch": {"F": float(welch.statistic), "df": (float(welch.df_num), float(welch.df_denom)), "p": float(welch.pvalue)}},
    )


# ---------------------------------------------------------------------------
# Repeated-measures ANOVA (helper)
# ---------------------------------------------------------------------------
from ..data import blocks  # noqa: E402
from ..result import CheckResult  # noqa: E402


@procedure("rm_anova", requires=("columns|outcome",))
def rm_anova(df: pd.DataFrame, spec: Spec) -> TestResult:
    from statsmodels.stats.anova import AnovaRM

    m = blocks(df, spec)
    if m.shape[1] < 3:
        raise SpecError("rm_anova needs at least three levels; use paired_t for two")
    n, k = m.shape
    subj = m.index.name or "subject"
    long = m.reset_index().rename(columns={m.index.name or "index": subj}).melt(id_vars=subj, var_name="time", value_name="y")
    long["time"] = long["time"].astype(str)
    res = AnovaRM(long, depvar="y", subject=subj, within=["time"]).fit()
    F, df1, df2, p = (float(res.anova_table.loc["time", c]) for c in ("F Value", "Num DF", "Den DF", "Pr > F"))

    cc = None
    extra: dict = {}
    sph = checks.manual("sphericity", "Sphericity (Mauchly's test is in pingouin)", "pingouin not installed")
    eff = None
    p_gg = None
    try:
        import pingouin as pg

        rm = pg.rm_anova(data=long, dv="y", within="time", subject=subj, correction=True, detailed=True)
        row = rm.iloc[0]
        cc = compare(
            "pingouin.rm_anova",
            "The uncorrected F should match AnovaRM. pingouin adds the Greenhouse–Geisser correction that statsmodels lacks.",
            {"F": F, "p": p},
            {"F": float(row["F"]), "p": float(row["p_unc"])},
        )
        p_gg = float(row["p_GG_corr"]) if "p_GG_corr" in rm.columns else None
        eps = float(row["eps"]) if "eps" in rm.columns else None
        ss_t, ss_e = float(row["SS"]), float(rm.iloc[1]["SS"])
        eff = Estimate("partial η²", ss_t / (ss_t + ss_e), note=f"generalized η² = {float(row['ng2']):.3f}" if "ng2" in rm.columns else "")
        sp = pg.sphericity(data=long, dv="y", within="time", subject=subj)
        sph = CheckResult("sphericity", "Sphericity (Mauchly's test)", bool(sp.spher), float(sp.W), float(sp.pval), f"Mauchly W = {sp.W:.3f}; Greenhouse–Geisser ε = {eps:.3f}" if eps is not None else f"Mauchly W = {sp.W:.3f}")
        extra = {"greenhouse_geisser": {"eps": eps, "p": p_gg}}
    except ImportError:
        pass
    # residuals of the additive subject + time model
    vals = m.to_numpy(float)
    resid = vals - vals.mean(1, keepdims=True) - vals.mean(0, keepdims=True) + vals.mean()
    dropped = m.attrs.get("dropped", 0)
    chk = [
        sph,
        CheckResult("balance", "Every subject has every level", bool(dropped == 0), None, None, f"{dropped} incomplete subject(s) dropped of {m.attrs.get('n_total', n)}" + ("; a mixed model (lmm) handles missing visits" if dropped else "")),
        checks.shapiro(resid.ravel(), "residual_normality", "Residuals roughly normal", spec.alpha),
    ]
    warnings = []
    if sph.passed is False and p_gg is not None:
        warnings.append(f"Sphericity violated: report the Greenhouse–Geisser corrected p = {p_gg:.4g}.")
    from itertools import combinations

    from statsmodels.stats.multitest import multipletests

    pairs = list(combinations(range(k), 2))
    ps = [stats.ttest_rel(vals[:, i], vals[:, j]).pvalue for i, j in pairs]
    adj = multipletests(ps, method="holm")[1]
    post = {"method": "pairwise paired t-tests, Holm-adjusted", "pairs": [{"a": str(m.columns[i]), "b": str(m.columns[j]), "p": float(pp), "p_adj": float(q)} for (i, j), pp, q in zip(pairs, ps, adj)]}
    return TestResult(
        test="rm_anova",
        n=int(n),
        statistic_name="F",
        statistic=F,
        p=p,
        df=(df1, df2),
        estimates=[Estimate(f"mean[{c}]", float(m[c].mean())) for c in m.columns],
        effect_size=eff,
        checks=chk,
        cross_check=cc,
        post_hoc=post,
        warnings=warnings,
        extra=extra,
    )
