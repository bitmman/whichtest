"""Categorical outcomes: binomial, 2×2, chi-square, goodness-of-fit, McNemar, Cochran's Q, Stuart–Maxwell."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable, Table, Table2x2, cochrans_q, mcnemar
from statsmodels.stats.proportion import proportion_confint

from .. import checks, effects
from ..data import blocks, column, contingency_table, int_param, validate_counts
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


def _k_n(df: pd.DataFrame | None, spec: Spec) -> tuple[int, int]:
    p = spec.params
    if "k" in p and "n" in p:
        k, n = int_param(p, "k"), int_param(p, "n", minimum=1)
        if k > n:
            raise SpecError("params.k cannot exceed params.n")
        return k, n
    if df is not None and spec.variables.outcome:
        y = column(df, spec.variables.outcome, "outcome")
        y = y[~np.isnan(y)]
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise SpecError("binom_test outcome must be coded 0/1")
        return int(y.sum()), int(len(y))
    raise SpecError("binom_test needs params.k and params.n, or a 0/1 variables.outcome column")


@procedure("binom_test", data_optional=True, directional=True)
def binom_test(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    k, n = _k_n(df, spec)
    p0 = float(spec.params.get("p0", 0.5))
    res = stats.binomtest(k, n, p=p0, alternative=spec.alternative)
    ci = res.proportion_ci(spec.ci_level, method="wilson")
    from statsmodels.stats.proportion import binom_test as sm_binom

    alt = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}[spec.alternative]
    p2 = sm_binom(k, n, prop=p0, alternative=alt)
    cc = compare(
        "statsmodels.stats.proportion.binom_test",
        "Same exact test in a second library.",
        {"p": res.pvalue},
        {"p": p2},
    )
    return TestResult(
        test="binom_test",
        n=n,
        statistic_name="successes",
        statistic=float(k),
        p=float(res.pvalue),
        estimates=[Estimate("proportion", k / n, (float(ci.low), float(ci.high)), spec.ci_level, f"Wilson; p₀ = {p0:g}")],
        effect_size=Estimate("difference from p₀", k / n - p0),
        checks=[checks.manual("independence", "Trials independent with a constant success probability")],
        cross_check=cc,
    )


@procedure("two_by_two", data_optional=True, directional=True)
def two_by_two(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    t, rows, cols = contingency_table(df, spec)
    if t.shape != (2, 2):
        raise SpecError(f"two_by_two needs a 2×2 table; got {t.shape}. Use chi_sq_indep for larger tables")
    exp = checks.expected_counts(t, "expected_counts", "Expected count ≥ 5 in every cell (else Fisher's exact)")
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    fisher = stats.fisher_exact(t, alternative=spec.alternative)
    directional = spec.alternative != "two-sided"
    # Fisher's exact test is the primary when cells are sparse, when asked for (params.exact), or when a
    # one-sided alternative is requested: the chi-square test has no direction.
    use_fisher = exp.passed is False or bool(spec.params.get("exact", False)) or directional
    tb = Table2x2(t.astype(int), shift_zeros=False)  # the Pearson statistic must use the observed counts
    r2 = tb.test_nominal_association()
    zero_cell = bool((t == 0).any())
    tb_ci = Table2x2(t.astype(int)) if zero_cell else tb  # a zero cell needs the 0.5 continuity shift for OR/RR intervals
    cc = compare(
        "statsmodels.stats.contingency_tables.Table2x2.test_nominal_association",
        "The Pearson chi-square should match scipy's uncorrected statistic exactly.",
        {"chi2": chi2, "p": p},
        {"chi2": r2.statistic, "p": r2.pvalue},
    )
    a = 1 - spec.ci_level
    or_ci = tb_ci.oddsratio_confint(alpha=a)
    rr_ci = tb_ci.riskratio_confint(alpha=a)
    shift_note = "; 0.5 added to every cell (zero cell)" if zero_cell else ""
    p1, p2_ = t[0, 0] / t[0].sum(), t[1, 0] / t[1].sum()
    rd = p1 - p2_
    se_rd = np.sqrt(p1 * (1 - p1) / t[0].sum() + p2_ * (1 - p2_) / t[1].sum())
    z = stats.norm.ppf(0.5 + spec.ci_level / 2)
    ests = [
        Estimate("odds ratio", float(tb_ci.oddsratio), (float(or_ci[0]), float(or_ci[1])), spec.ci_level, "" + shift_note.lstrip("; ")),
        Estimate("risk ratio", float(tb_ci.riskratio), (float(rr_ci[0]), float(rr_ci[1])), spec.ci_level, "not valid for case–control sampling" + shift_note),
        Estimate("risk difference", float(rd), (float(rd - z * se_rd), float(rd + z * se_rd)), spec.ci_level, "Wald"),
    ]
    warnings = []
    if use_fisher:
        why = "One-sided alternative requested" if directional and exp.passed is not False else ("Requested (params.exact)" if exp.passed is not False else "Small expected counts")
        warnings.append(f"{why}: reporting Fisher's exact p as the primary p-value; the (two-sided) chi-square is kept in extra.")
    return TestResult(
        test="two_by_two",
        n=int(t.sum()),
        statistic_name="Fisher odds ratio" if use_fisher else "Pearson χ²",
        statistic=float(fisher.statistic) if use_fisher else float(chi2),
        p=float(fisher.pvalue) if use_fisher else float(p),
        df=None if use_fisher else 1.0,
        estimates=ests,
        effect_size=effects.cramers_v(t),
        checks=[exp],
        cross_check=cc,
        warnings=warnings,
        extra={"table": t.tolist(), "rows": [str(r) for r in rows], "cols": [str(c) for c in cols], "chi2": {"statistic": float(chi2), "p": float(p)}, "fisher": {"odds_ratio": float(fisher.statistic), "p": float(fisher.pvalue)}},
    )


@procedure("chi_sq_indep", data_optional=True)
def chi_sq_indep(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    t, rows, cols = contingency_table(df, spec)
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    r2 = Table(t, shift_zeros=False).test_nominal_association()  # same observed counts as scipy, zeros included
    cc = compare(
        "statsmodels.stats.contingency_tables.Table.test_nominal_association",
        "Second implementation. The statistic and p-value should match.",
        {"chi2": chi2, "p": p, "df": dof},
        {"chi2": r2.statistic, "p": r2.pvalue, "df": r2.df},
    )
    exp = checks.expected_counts(t, "expected_counts", "Expected count ≥ 5 in at least 80% of cells, none < 1")
    resid = np.asarray(Table(t, shift_zeros=False).standardized_resids)
    warnings = []
    extra = {"table": t.tolist(), "rows": [str(r) for r in rows], "cols": [str(c) for c in cols], "standardized_resids": resid.tolist()}
    if exp.passed is False:
        # permutation test of the chi-square statistic (the taxonomy's fallback technique)
        rng = np.random.default_rng(int(spec.params.get("seed", 0)))
        n_perm = int(spec.params.get("n_perm", 2000))
        ri = np.repeat(np.arange(t.shape[0]), t.sum(1).astype(int))
        ci = np.repeat(np.arange(t.shape[1]), t.sum(0).astype(int))
        count = 0
        for _ in range(n_perm):
            tt = np.zeros_like(t)
            np.add.at(tt, (ri, rng.permutation(ci)), 1)
            e = np.outer(tt.sum(1), tt.sum(0)) / tt.sum()
            s = ((tt - e) ** 2 / e).sum()
            count += s >= chi2 - 1e-12
        p_perm = (count + 1) / (n_perm + 1)
        extra["permutation_p"] = float(p_perm)
        warnings.append(f"Sparse cells: permutation p = {p_perm:.4g} ({n_perm} permutations); prefer it over the asymptotic p.")
    return TestResult(
        test="chi_sq_indep",
        n=int(t.sum()),
        statistic_name="Pearson χ²",
        statistic=float(chi2),
        p=float(p),
        df=float(dof),
        effect_size=effects.cramers_v(t),
        checks=[exp],
        cross_check=cc,
        post_hoc={"method": "adjusted standardized residuals per cell (|r| > 2 flags a cell)", "resids": resid.tolist()},
        warnings=warnings,
        extra=extra,
    )


@procedure("chi_gof", data_optional=True)
def chi_gof(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    p = spec.params
    if "observed" in p:
        obs = np.asarray(p["observed"], float)
        labels = [str(i) for i in range(len(obs))]
    elif df is not None and spec.variables.outcome:
        vc = df[spec.variables.outcome].value_counts().sort_index()
        obs, labels = vc.to_numpy(float), [str(i) for i in vc.index]
    else:
        raise SpecError("chi_gof needs params.observed (counts) or a categorical variables.outcome column")
    exp = p.get("expected")
    if exp is None:
        exp = np.full(len(obs), obs.sum() / len(obs))
    else:
        exp = np.asarray(exp, float)
        if np.isclose(exp.sum(), 1.0):
            exp = exp * obs.sum()
        elif not np.isclose(exp.sum(), obs.sum()):
            raise SpecError("params.expected must sum to the observed total or to 1")
    res = stats.chisquare(f_obs=obs, f_exp=exp)
    g = stats.power_divergence(obs, exp, lambda_="log-likelihood")
    cc = compare(
        "G-test: scipy.stats.power_divergence(lambda_='log-likelihood')",
        "The G-test answers the same question. Results should be close in large samples.",
        {"p": res.pvalue},
        {"p": g.pvalue},
        kind="approximate",
        alpha=spec.alpha,
        detail=f"G = {g.statistic:.3f}",
    )
    chk = CheckResult("expected_counts", "Expected count ≥ 5 per category", bool(exp.min() >= 5), float(exp.min()), None, f"min expected = {exp.min():.2f}")
    warnings = []
    if chk.passed is False and len(obs) <= 6 and obs.sum() <= 200:
        from scipy.stats import multinomial

        # exact multinomial test by enumeration is exponential; use Monte Carlo
        rng = np.random.default_rng(int(p.get("seed", 0)))
        probs = exp / exp.sum()
        sims = rng.multinomial(int(obs.sum()), probs, size=int(p.get("n_sim", 5000)))
        s = ((sims - exp) ** 2 / exp).sum(1)
        p_mc = float(((s >= res.statistic - 1e-12).sum() + 1) / (len(s) + 1))
        warnings.append(f"Small expected counts: Monte-Carlo exact multinomial p = {p_mc:.4g}.")
    return TestResult(
        test="chi_gof",
        n=int(obs.sum()),
        statistic_name="χ²",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(len(obs) - 1),
        estimates=[Estimate(f"proportion[{l}]", float(o / obs.sum()), note=f"expected {e / obs.sum():.3f}") for l, o, e in zip(labels, obs, exp)],
        effect_size=effects.cohen_w(float(res.statistic), int(obs.sum())),
        checks=[chk],
        cross_check=cc,
        warnings=warnings,
    )


def _sorted_levels(values) -> list:
    """Category levels in a stable order: numeric order when all numeric, else string order."""
    levels = list(pd.unique(np.asarray(list(values), dtype=object)))
    try:
        return sorted(levels, key=float)
    except (TypeError, ValueError):
        return sorted(levels, key=str)


def _paired_table(df: pd.DataFrame | None, spec: Spec) -> np.ndarray:
    if "table" in spec.params:
        t = validate_counts(spec.params["table"], "params.table")
        if t.ndim != 2:
            raise SpecError("params.table must be a 2-D array of counts")
        return t
    if df is None:
        raise SpecError(f"{spec.test} needs params.table or paired columns")
    from ..data import paired

    a, b, _ = paired(df, spec, numeric=False)  # nominal labels stay labels
    cats = _sorted_levels(list(a) + list(b))
    idx = {c: i for i, c in enumerate(cats)}
    t = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        t[idx[x], idx[y]] += 1
    return t


@procedure("mcnemar", data_optional=True)
def mcnemar_test(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    t = _paired_table(df, spec)
    if t.shape != (2, 2):
        raise SpecError(f"mcnemar needs a 2×2 paired table; got {t.shape}. Use stuart_maxwell for more categories")
    b, c = int(t[0, 1]), int(t[1, 0])
    exact = bool(spec.params.get("exact", b + c < 25))
    res = mcnemar(t, exact=exact, correction=True)
    if exact:
        p2 = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else float("nan")
        cc = compare(
            "scipy.stats.binomtest(b, b + c, 0.5)",
            "Exact McNemar is a binomial test on the discordant pairs. The p-value is identical.",
            {"p": res.pvalue},
            {"p": p2},
        )
    else:
        p2 = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else float("nan")
        cc = compare(
            "scipy.stats.binomtest(b, b + c, 0.5) (exact)",
            "The exact binomial p and the continuity-corrected chi-square p should agree on the decision.",
            {"p": res.pvalue},
            {"p": p2},
            kind="approximate",
            alpha=spec.alpha,
        )
    n = int(t.sum())
    p1, p2_ = t[0].sum() / n, t[:, 0].sum() / n
    diff = p1 - p2_
    se = np.sqrt((b + c) - (b - c) ** 2 / n) / n if n > 0 else float("nan")
    z = stats.norm.ppf(0.5 + spec.ci_level / 2)
    por = b / c if c > 0 else float("inf")
    por_ci = None
    if b > 0 and c > 0:
        lse = np.sqrt(1 / b + 1 / c)
        por_ci = (float(np.exp(np.log(por) - z * lse)), float(np.exp(np.log(por) + z * lse)))
    return TestResult(
        test="mcnemar",
        n=n,
        statistic_name="discordant b" if exact else "McNemar χ² (corrected)",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=None if exact else 1.0,
        estimates=[
            Estimate("difference in marginal proportions", float(diff), (float(diff - z * se), float(diff + z * se)), spec.ci_level, "Wald"),
            Estimate("paired odds ratio b/c", float(por), por_ci, spec.ci_level if por_ci else None),
        ],
        checks=[
            checks.manual("paired_table", "Rows = first measurement, columns = second, same subjects"),
            CheckResult("discordant_count", "Enough discordant pairs for the test to say anything", bool(b + c >= 10), float(b + c), None, f"b + c = {b + c}"),
        ],
        cross_check=cc,
        extra={"table": t.tolist(), "exact": exact},
    )


@procedure("cochran_q", requires=("columns|outcome",))
def cochran_q_test(df: pd.DataFrame, spec: Spec) -> TestResult:
    m = blocks(df, spec)
    x = m.to_numpy(float)
    if not set(np.unique(x)) <= {0.0, 1.0}:
        raise SpecError("cochran_q needs 0/1 outcomes")
    if m.shape[1] < 3:
        raise SpecError("cochran_q needs at least three conditions; use mcnemar for two")
    res = cochrans_q(x)
    cc = None
    try:
        import pingouin as pg

        subj = m.index.name or "index"
        long = m.reset_index().melt(id_vars=subj, var_name="cond", value_name="y")
        pc = pg.cochran(data=long, dv="y", within="cond", subject=subj)
        cc = compare(
            "pingouin.cochran",
            "Second implementation. Q and p should match.",
            {"Q": res.statistic, "p": res.pvalue},
            {"Q": float(pc["Q"].iloc[0]), "p": float(pc[checks.pg_col(pc, "p_unc")].iloc[0])},
        )
    except ImportError:
        pass
    n, k = m.shape
    ests = []
    for c in m.columns:
        s = int(m[c].sum())
        lo, hi = proportion_confint(s, n, alpha=1 - spec.ci_level, method="wilson")
        ests.append(Estimate(f"proportion[{c}]", s / n, (float(lo), float(hi)), spec.ci_level, "Wilson"))
    pairs = list(combinations(range(k), 2))
    ps = []
    for i, j in pairs:
        a, b = x[:, i], x[:, j]
        bb, cc_ = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
        ps.append(stats.binomtest(bb, bb + cc_, 0.5).pvalue if bb + cc_ > 0 else 1.0)
    from statsmodels.stats.multitest import multipletests

    adj = multipletests(ps, method="holm")[1] if ps else []
    dropped = m.attrs.get("dropped", 0)
    return TestResult(
        test="cochran_q",
        n=int(n),
        statistic_name="Q",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(res.df),
        estimates=ests,
        checks=[CheckResult("complete_blocks", "Every subject observed under every condition", bool(dropped == 0), None, None, f"{dropped} incomplete subject(s) dropped")],
        cross_check=cc,
        post_hoc={"method": "pairwise exact McNemar, Holm-adjusted", "pairs": [{"a": str(m.columns[i]), "b": str(m.columns[j]), "p": float(p), "p_adj": float(q)} for (i, j), p, q in zip(pairs, ps, adj)]},
    )


@procedure("stuart_maxwell", data_optional=True)
def stuart_maxwell(df: pd.DataFrame | None, spec: Spec) -> TestResult:
    t = _paired_table(df, spec)
    if t.shape[0] != t.shape[1]:
        raise SpecError("stuart_maxwell needs a square table (same categories at both times)")
    sq = SquareTable(t, shift_zeros=False)
    r = sq.homogeneity(method="stuart_maxwell")
    r2 = sq.homogeneity(method="bhapkar")
    cc = compare(
        "SquareTable.homogeneity(method='bhapkar')",
        "Bhapkar's test targets the same hypothesis and is asymptotically equivalent.",
        {"p": r.pvalue},
        {"p": r2.pvalue},
        kind="approximate",
        alpha=spec.alpha,
        detail=f"Bhapkar χ² = {float(r2.statistic):.3f}",
    )
    n = t.sum()
    ests = [Estimate(f"marginal[{i}] first vs second", float(t[i].sum() / n), note=f"second: {t[:, i].sum() / n:.3f}") for i in range(t.shape[0])]
    return TestResult(
        test="stuart_maxwell",
        n=int(n),
        statistic_name="Stuart–Maxwell χ²",
        statistic=float(r.statistic),
        p=float(r.pvalue),
        df=float(r.df),
        estimates=ests,
        checks=[CheckResult("square_table", "Same categories at both measurements", True, detail=f"{t.shape[0]} categories")],
        cross_check=cc,
        extra={"table": t.tolist()},
    )
