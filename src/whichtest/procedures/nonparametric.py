"""Rank-based tests: Wilcoxon (one-sample and paired), Mann–Whitney, Kruskal–Wallis, Friedman."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import checks, effects
from ..data import blocks, column, groups, paired, float_param
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


def _sign_test(d: np.ndarray, alternative: str) -> tuple[float, float]:
    nz = d[d != 0]
    k, n = int((nz > 0).sum()), int(len(nz))
    if n == 0:
        return float("nan"), float("nan")
    r = stats.binomtest(k, n, 0.5, alternative=alternative)
    return float(r.pvalue), k / n - 0.5


def _symmetry_check(d: np.ndarray) -> CheckResult:
    d = d[d != 0]
    if len(d) < 5:
        return CheckResult("symmetry", "Differences symmetric about the centre", None, detail="too few non-zero values")
    sk = float(stats.skew(d))
    return CheckResult(
        "symmetry",
        "Differences symmetric about the centre",
        bool(abs(sk) < 1.0),
        sk,
        None,
        f"skew of non-zero differences = {sk:.2f} (|skew| ≥ 1 is a red flag)",
    )


def _wilcoxon_core(d: np.ndarray, spec: Spec, test: str, labels: str) -> TestResult:
    zero_method = spec.params.get("zero_method", "wilcox")
    res = stats.wilcoxon(d, zero_method=zero_method, alternative=spec.alternative)
    p_sign, dir_sign = _sign_test(d, spec.alternative)
    dir_w = float(np.median(d[d != 0])) if (d != 0).any() else 0.0
    cc = compare(
        "sign test: scipy.stats.binomtest on the sign of the differences",
        "The sign test has less power but should point the same way. If it doesn't, look at the distribution of differences.",
        {"direction": dir_w, "p": res.pvalue},
        {"direction": dir_sign, "p": p_sign},
        kind="direction",
        detail=f"sign test p = {p_sign:.4g}",
    )
    n_zero = int((d == 0).sum())
    chk = [_symmetry_check(d), checks.note("zero_handling", f"zero_method={zero_method!r}", f"{n_zero} zero difference(s)")]
    hl = effects.hodges_lehmann_paired(d, spec.ci_level) if test == "wilcoxon_signed" else effects.hodges_lehmann_one(d, spec.ci_level)
    return TestResult(
        test=test,
        n=int(len(d)),
        statistic_name="W",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        estimates=[hl, Estimate(f"median {labels}", dir_w)],
        effect_size=effects.rank_biserial_paired(d),
        checks=chk,
        cross_check=cc,
    )


@procedure("wilcoxon_signed", requires=("before|columns|outcome",), directional=True)
def wilcoxon_signed(df: pd.DataFrame, spec: Spec) -> TestResult:
    a, b, (la, lb) = paired(df, spec)
    return _wilcoxon_core(b - a, spec, "wilcoxon_signed", f"difference ({lb} − {la})")


@procedure("one_sample_wilcoxon", requires=("outcome",), directional=True)
def one_sample_wilcoxon(df: pd.DataFrame, spec: Spec) -> TestResult:
    x = column(df, spec.variables.outcome, "outcome")
    x = x[~np.isnan(x)]
    m0 = float_param(spec.params, "m0" if "m0" in spec.params else "mu0", default=0.0)
    r = _wilcoxon_core(x - m0, spec, "one_sample_wilcoxon", f"(x − {m0:g})")
    r.estimates[0].name = f"Hodges–Lehmann pseudo-median of x − {m0:g}"
    return r


@procedure("mann_whitney", requires=("outcome", "group"), directional=True)
def mann_whitney(df: pd.DataFrame, spec: Spec) -> TestResult:
    g = groups(df, spec.variables.outcome, spec.variables.group)
    if len(g) != 2:
        raise SpecError(f"mann_whitney needs exactly two groups; {spec.variables.group!r} has {len(g)}: {list(g)}")
    (la, a), (lb, b) = g.items()
    res = stats.mannwhitneyu(a, b, alternative=spec.alternative)
    z, p2 = stats.ranksums(a, b, alternative=spec.alternative)
    cc = compare(
        "scipy.stats.ranksums",
        "The rank-sum test is the same test with a normal approximation. p-values should be close in moderate samples.",
        {"p": res.pvalue},
        {"p": p2},
        kind="approximate",
        alpha=spec.alpha,
        detail=f"rank-sum z = {z:.3f}",
    )
    rb, cl = effects.rank_biserial_indep(float(res.statistic), len(a), len(b))
    sd_ratio = float(np.std(a, ddof=1) / np.std(b, ddof=1)) if np.std(b, ddof=1) > 0 else float("nan")
    shape = CheckResult(
        "shape_similarity",
        "Similar distribution shapes if you want to talk about medians. Otherwise it tests stochastic dominance",
        None if np.isnan(sd_ratio) else bool(0.5 <= sd_ratio <= 2.0),
        sd_ratio,
        None,
        f"SD ratio {la}/{lb} = {sd_ratio:.2f}; skew {stats.skew(a):.2f} vs {stats.skew(b):.2f}",
    )
    warnings = []
    if shape.passed is False:
        bm = stats.brunnermunzel(a, b, alternative=spec.alternative)
        warnings.append(f"Spreads differ a lot: Brunner–Munzel gives W = {bm.statistic:.3f}, p = {bm.pvalue:.4g}.")
    return TestResult(
        test="mann_whitney",
        n={str(la): int(len(a)), str(lb): int(len(b))},
        statistic_name="U",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        estimates=[Estimate(f"median[{la}]", float(np.median(a))), Estimate(f"median[{lb}]", float(np.median(b))), cl],
        effect_size=rb,
        checks=[shape],
        cross_check=cc,
        warnings=warnings,
    )


@procedure("kruskal", requires=("outcome", "group"))
def kruskal(df: pd.DataFrame, spec: Spec) -> TestResult:
    g = groups(df, spec.variables.outcome, spec.variables.group)
    arrays = list(g.values())
    res = stats.kruskal(*arrays)
    y = np.concatenate(arrays)
    lab = np.concatenate([[str(k)] * len(v) for k, v in g.items()])
    r = stats.rankdata(y)
    f, p2 = stats.f_oneway(*[r[lab == str(k)] for k in g])
    cc = compare(
        "one-way ANOVA on ranks (scipy.stats.f_oneway on rankdata)",
        "ANOVA on ranks approximates Kruskal–Wallis. It should agree on direction and roughly on p.",
        {"p": res.pvalue},
        {"p": p2},
        kind="approximate",
        alpha=spec.alpha,
        detail=f"rank ANOVA F = {f:.3f}",
    )
    sds = np.array([np.std(v, ddof=1) for v in arrays])
    ratio = float(sds.max() / sds.min()) if sds.min() > 0 else float("nan")
    shape = CheckResult(
        "shape_similarity",
        "Similar distribution shapes across groups",
        None if np.isnan(ratio) else bool(ratio <= 2.0),
        ratio,
        None,
        f"max/min SD ratio = {ratio:.2f}",
    )
    post = None
    if len(g) >= 3:
        try:
            import scikit_posthocs as sp

            d = pd.DataFrame({"y": y, "group": lab})
            dunn = sp.posthoc_dunn(d, val_col="y", group_col="group", p_adjust="holm")
            post = {"method": "Dunn's test, Holm-adjusted (scikit_posthocs.posthoc_dunn)", "p_adj": dunn.to_dict()}
        except ImportError:
            post = {"method": "Dunn's test needs scikit-posthocs (pip install whichtest[cross])"}
    return TestResult(
        test="kruskal",
        n={str(k): int(len(v)) for k, v in g.items()},
        statistic_name="H",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(len(g) - 1),
        estimates=[Estimate(f"median[{k}]", float(np.median(v))) for k, v in g.items()],
        effect_size=effects.epsilon_sq(float(res.statistic), len(y)),
        checks=[shape],
        cross_check=cc,
        post_hoc=post,
    )


@procedure("friedman", requires=("columns|outcome",))
def friedman(df: pd.DataFrame, spec: Spec) -> TestResult:
    m = blocks(df, spec)
    if m.shape[1] < 3:
        raise SpecError("friedman needs at least three conditions; use wilcoxon_signed for two")
    cols = [m[c].to_numpy(float) for c in m.columns]
    res = stats.friedmanchisquare(*cols)
    n, k = m.shape
    kw = effects.kendall_w(float(res.statistic), n, k)

    cc = None
    try:
        import pingouin as pg

        long = m.reset_index().melt(id_vars=m.index.name or "index", var_name="cond", value_name="y")
        subj = m.index.name or "index"
        pf = pg.friedman(data=long, dv="y", within="cond", subject=subj)
        cc = compare(
            "pingouin.friedman",
            "Same statistic. pingouin also reports Kendall's W.",
            {"Q": res.statistic, "p": res.pvalue, "W": kw.value},
            {"Q": float(pf["Q"].iloc[0]), "p": float(pf[checks.pg_col(pf, "p_unc")].iloc[0]), "W": float(pf["W"].iloc[0])},
        )
    except ImportError:
        pass

    dropped = m.attrs.get("dropped", 0)
    chk = [
        CheckResult(
            "complete_blocks",
            "Every subject measured under every condition",
            bool(dropped == 0),
            None,
            None,
            f"{dropped} incomplete subject(s) dropped of {m.attrs.get('n_total', n)}",
        )
    ]

    post = None
    if k >= 3:
        from itertools import combinations

        from statsmodels.stats.multitest import multipletests

        pairs = list(combinations(range(k), 2))
        ps = [stats.wilcoxon(cols[i], cols[j]).pvalue for i, j in pairs]
        adj = multipletests(ps, method="holm")[1]
        post = {
            "method": "pairwise Wilcoxon signed-rank, Holm-adjusted",
            "pairs": [{"a": str(m.columns[i]), "b": str(m.columns[j]), "p": float(p), "p_adj": float(q)} for (i, j), p, q in zip(pairs, ps, adj)],
        }
    return TestResult(
        test="friedman",
        n=int(n),
        statistic_name="χ²_F",
        statistic=float(res.statistic),
        p=float(res.pvalue),
        df=float(k - 1),
        estimates=[Estimate(f"median[{c}]", float(np.median(m[c]))) for c in m.columns],
        effect_size=kw,
        checks=chk,
        cross_check=cc,
        post_hoc=post,
    )
