"""Agreement between raters: Cohen's kappa, Fleiss' kappa."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, cohens_kappa, fleiss_kappa, to_table

from .. import checks, effects
from ..data import blocks
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError


@procedure("cohen_kappa", requires=("columns|outcome",))
def cohen_kappa(df: pd.DataFrame, spec: Spec) -> TestResult:
    m = blocks(df, spec)
    if m.shape[1] != 2:
        raise SpecError("cohen_kappa needs exactly two raters; use fleiss_kappa for more")
    r1, r2 = m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy()
    wt = spec.params.get("weights")  # None | 'linear' | 'quadratic'
    cats = sorted(set(np.unique(r1)) | set(np.unique(r2)))
    idx = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for a, b in zip(r1, r2):
        table[idx[a], idx[b]] += 1
    res = cohens_kappa(table, wt=wt, return_results=True)
    cc = None
    try:
        from sklearn.metrics import cohen_kappa_score

        k2 = cohen_kappa_score(r1, r2, weights=wt)
        cc = compare(
            "sklearn.metrics.cohen_kappa_score",
            "Point estimates should match. statsmodels adds the standard error and CI.",
            {"kappa": res.kappa},
            {"kappa": k2},
        )
    except ImportError:
        pass
    agree = float((r1 == r2).mean())
    prev = table.sum(1) / table.sum()
    from scipy import stats as st

    z = st.norm.ppf(0.5 + spec.ci_level / 2)
    ci = (float(res.kappa - z * res.std_kappa), float(res.kappa + z * res.std_kappa))
    return TestResult(
        test="cohen_kappa",
        n=int(len(r1)),
        statistic_name="z (κ = 0)",
        statistic=float(res.z_value),
        p=float(res.pvalue_two_sided),
        effect_size=Estimate(f"{'weighted ' if wt else ''}κ", float(res.kappa), ci, spec.ci_level, f"weights = {wt}"),
        estimates=[Estimate("percent agreement", agree)],
        checks=[
            CheckResult("same_categories", "Both raters use the same category set", bool(set(np.unique(r1)) == set(np.unique(r2))), detail=f"categories: {[str(c) for c in cats]}"),
            CheckResult("prevalence", "No category dominates (κ is depressed by skewed prevalence)", bool(prev.max() < 0.8), float(prev.max()), None, f"max category prevalence (rater 1) = {prev.max():.2f}"),
        ],
        cross_check=cc,
        extra={"table": table.tolist(), "categories": [str(c) for c in cats]},
    )


@procedure("fleiss_kappa", requires=("columns|outcome",))
def fleiss(df: pd.DataFrame, spec: Spec) -> TestResult:
    m = blocks(df, spec)
    if m.shape[1] < 2:
        raise SpecError("fleiss_kappa needs at least two raters")
    ratings = m.to_numpy()
    counts, cats = aggregate_raters(ratings)
    k = float(fleiss_kappa(counts))
    cc = None
    try:
        import krippendorff

        alpha = float(krippendorff.alpha(reliability_data=ratings.T.astype(float), level_of_measurement="nominal"))
        cc = compare(
            "krippendorff.alpha (nominal)",
            "Krippendorff's α is close to Fleiss' κ for complete nominal data, and also handles missing ratings.",
            {"agreement": k},
            {"agreement": alpha},
            kind="exact",
            rtol=0.0,
            atol=0.05,
            detail=f"α = {alpha:.4f}; |κ − α| within 0.05 counts as agreement",
        )
    except ImportError:
        pass
    n_boot = int(spec.params.get("n_boot", 1000))
    lo, hi = effects.bootstrap_ci(lambda r: fleiss_kappa(aggregate_raters(r)[0]), ratings, level=spec.ci_level, n_boot=n_boot, seed=int(spec.params.get("seed", 0)))
    dropped = m.attrs.get("dropped", 0)
    return TestResult(
        test="fleiss_kappa",
        n=int(ratings.shape[0]),
        statistic_name="",
        statistic=None,
        p=None,
        effect_size=Estimate("Fleiss' κ", k, (lo, hi), spec.ci_level, f"bootstrap, {n_boot} resamples"),
        estimates=[Estimate("raters", float(ratings.shape[1])), Estimate("categories", float(len(cats)))],
        checks=[CheckResult("complete_ratings", "Every item rated by every rater", bool(dropped == 0), None, None, f"{dropped} incomplete item(s) dropped")],
        cross_check=cc,
        extra={"categories": [str(c) for c in cats]},
    )


# ---------------------------------------------------------------------------
# Method comparison and diagnostic accuracy (helper tests)
# ---------------------------------------------------------------------------
from scipy import stats as _st  # noqa: E402

from ..data import column, paired  # noqa: E402
from ..delong import delong_ci, delong_paired  # noqa: E402


def _truth(df: pd.DataFrame, spec: Spec, role: str = "outcome") -> np.ndarray:
    """0/1 truth labels; ``params.positive_label`` names the positive class explicitly."""
    raw = column(df, getattr(spec.variables, role), role, numeric=False)
    pos = spec.params.get("positive_label")
    levels = [l for l in pd.unique(raw) if not pd.isna(l)]
    if pos is None:
        if set(levels) <= {0, 1, 0.0, 1.0, True, False}:
            pos = 1
        else:
            raise SpecError(f"set params.positive_label: {getattr(spec.variables, role)!r} has levels {levels}")
    if pos not in levels and str(pos) not in [str(l) for l in levels]:
        raise SpecError(f"positive_label {pos!r} not among levels {levels}")
    return (raw.astype(str) == str(pos)).astype(int)


@procedure("bland_altman", requires=("columns|before|outcome",))
def bland_altman(df: pd.DataFrame, spec: Spec) -> TestResult:
    m1, m2, (l1, l2) = paired(df, spec)
    d = m1 - m2
    mean = (m1 + m2) / 2
    n = len(d)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    z = _st.norm.ppf(0.5 + spec.ci_level / 2)
    k = float(spec.params.get("loa_multiplier", 1.96))
    loa = (bias - k * sd, bias + k * sd)
    t = _st.t.ppf(0.5 + spec.ci_level / 2, n - 1)
    se_bias = sd / np.sqrt(n)
    se_loa = sd * np.sqrt(1 / n + k**2 / (2 * (n - 1)))  # Bland & Altman 1986/1999
    # cross-check: statsmodels' plot draws the same three lines
    cc = None
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import statsmodels.api as sm

        fig = sm.graphics.mean_diff_plot(m1, m2, sd_limit=k)
        ys = sorted({float(np.unique(line.get_ydata())[0]) for line in fig.axes[0].get_lines() if len(np.unique(line.get_ydata())) == 1})
        plt.close(fig)
        sd0 = float(d.std(ddof=0))  # statsmodels draws its limits with the population SD
        cc = compare(
            "statsmodels.graphics.mean_diff_plot line positions",
            "The plot draws its own bias and limit lines. They should sit exactly where the hand-computed values say.",
            {"lower_loa": bias - k * sd0, "bias": bias, "upper_loa": bias + k * sd0},
            {"lower_loa": ys[0] if len(ys) == 3 else None, "bias": ys[1] if len(ys) == 3 else None, "upper_loa": ys[2] if len(ys) == 3 else None},
            rtol=1e-9,
            detail="compared with ddof=0 limits, which is what the plot draws; the reported limits use ddof=1",
        )
    except ImportError:
        pass
    import statsmodels.api as sm

    reg = sm.OLS(d, sm.add_constant(mean)).fit()
    slope_p = float(reg.pvalues[1])
    chk = [
        checks.shapiro(d, "difference_normality", "Differences roughly normal", spec.alpha),
        CheckResult("proportional_bias", "No proportional bias: regress differences on means", bool(slope_p >= spec.alpha), float(reg.params[1]), slope_p, f"slope of difference on mean = {reg.params[1]:.4f}"),
    ]
    warnings = []
    if chk[1].passed is False:
        warnings.append("Differences grow with magnitude: consider log-transforming or reporting ratio limits.")
    outside = int(((d < loa[0]) | (d > loa[1])).sum())
    return TestResult(
        test="bland_altman",
        n=n,
        statistic_name="",
        statistic=None,
        p=None,
        estimates=[
            Estimate(f"bias ({l1} − {l2})", bias, (bias - t * se_bias, bias + t * se_bias), spec.ci_level),
            Estimate("lower limit of agreement", loa[0], (loa[0] - t * se_loa, loa[0] + t * se_loa), spec.ci_level),
            Estimate("upper limit of agreement", loa[1], (loa[1] - t * se_loa, loa[1] + t * se_loa), spec.ci_level),
            Estimate("SD of differences", sd),
            Estimate("points outside the limits", float(outside), note=f"{outside / n:.1%}; expect about 5%"),
        ],
        checks=chk,
        cross_check=cc,
        warnings=warnings + ["Correlation measures association, not agreement; judge the limits against what is clinically acceptable."],
    )


@procedure("diag_binary", requires=("outcome", "x"))
def diag_binary(df: pd.DataFrame, spec: Spec) -> TestResult:
    truth = _truth(df, spec, "outcome")
    test_raw = column(df, spec.variables.x, "x", numeric=False)
    pos_t = spec.params.get("test_positive_label")
    if pos_t is None:
        t_levels = [l for l in pd.unique(test_raw) if not pd.isna(l)]
        if set(t_levels) <= {0, 1, 0.0, 1.0, True, False}:
            pos_t = 1
        else:
            raise SpecError(f"set params.test_positive_label: {spec.variables.x!r} has levels {t_levels}")
    test = (test_raw.astype(str) == str(pos_t)).astype(int)
    from sklearn.metrics import confusion_matrix
    from statsmodels.stats.proportion import proportion_confint

    tn, fp, fn, tp = confusion_matrix(truth, test, labels=[0, 1]).ravel()
    ct = pd.crosstab(truth, test).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    cc = compare(
        "pandas.crosstab(truth, test)",
        "An independent tally catches label-encoding mistakes about which class is positive, the most common bug here.",
        {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        {"tp": ct.loc[1, 1], "fp": ct.loc[0, 1], "fn": ct.loc[1, 0], "tn": ct.loc[0, 0]},
    )
    a = 1 - spec.ci_level

    def prop(k: int, n: int, name: str, note: str = "") -> Estimate:
        if n == 0:
            return Estimate(name, None, note="no cases")
        lo, hi = proportion_confint(k, n, alpha=a, method="wilson")
        return Estimate(name, k / n, (float(lo), float(hi)), spec.ci_level, note or f"{k}/{n}")

    sens, spc = tp / (tp + fn) if tp + fn else float("nan"), tn / (tn + fp) if tn + fp else float("nan")
    z = _st.norm.ppf(0.5 + spec.ci_level / 2)

    def lr(num_k, num_n, den_k, den_n, name):
        if min(num_n, den_n) == 0 or den_k == 0 or num_k == 0:
            return Estimate(name, None, note="undefined (a zero cell)")
        val = (num_k / num_n) / (den_k / den_n)
        se = np.sqrt(1 / num_k - 1 / num_n + 1 / den_k - 1 / den_n)
        return Estimate(name, float(val), (float(np.exp(np.log(val) - z * se)), float(np.exp(np.log(val) + z * se))), spec.ci_level, "log method")

    prev = truth.mean()
    ests = [
        prop(tp, tp + fn, "sensitivity"),
        prop(tn, tn + fp, "specificity"),
        prop(tp, tp + fp, "PPV", "depends on prevalence"),
        prop(tn, tn + fn, "NPV", "depends on prevalence"),
        lr(tp, tp + fn, fp, fp + tn, "LR+"),
        lr(fn, tp + fn, tn, fp + tn, "LR−"),
        prop(tp + tn, tp + tn + fp + fn, "accuracy"),
        Estimate("prevalence in sample", float(prev)),
    ]
    return TestResult(
        test="diag_binary",
        n=int(len(truth)),
        statistic_name="",
        statistic=None,
        p=None,
        estimates=ests,
        effect_size=Estimate("Youden's J", float(sens + spc - 1) if np.isfinite(sens + spc) else None),
        checks=[
            checks.manual("verification", "Gold standard applied to everyone, not only to test positives"),
            checks.manual("prevalence", "PPV and NPV depend on prevalence. Skip them in case–control samples", f"sample prevalence = {prev:.1%}"),
        ],
        cross_check=cc,
        extra={"table": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}, "positive_label": spec.params.get("positive_label", 1)},
    )


@procedure("roc_auc", requires=("outcome", "x"))
def roc_auc(df: pd.DataFrame, spec: Spec) -> TestResult:
    truth = _truth(df, spec, "outcome")
    score = column(df, spec.variables.x, "x")
    keep = ~np.isnan(score)
    truth, score = truth[keep], score[keep]
    if truth.sum() == 0 or truth.sum() == len(truth):
        raise SpecError("roc_auc needs both positive and negative cases")
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(truth, score))
    n1, n0 = int(truth.sum()), int(len(truth) - truth.sum())
    u = _st.mannwhitneyu(score[truth == 1], score[truth == 0]).statistic
    cc = compare(
        "scipy.stats.mannwhitneyu U / (n₁ n₀)",
        "AUC equals the Mann–Whitney U divided by n₁n₀. They must match exactly; if not, the positive class is flipped.",
        {"auc": auc},
        {"auc": u / (n1 * n0)},
        rtol=1e-10,
    )
    auc_d, ci, se = delong_ci(truth, score, spec.ci_level)
    p = float(2 * _st.norm.sf(abs((auc - 0.5) / se))) if se > 0 else None
    fpr, tpr, thr = roc_curve(truth, score)
    ests = [Estimate("AUC", auc, ci, spec.ci_level, "DeLong")]
    thr_fixed = spec.params.get("threshold")
    warnings = []
    if thr_fixed is not None:
        pred = (score >= float(thr_fixed)).astype(int)
        tp, fn = int(((pred == 1) & (truth == 1)).sum()), int(((pred == 0) & (truth == 1)).sum())
        tn, fp = int(((pred == 0) & (truth == 0)).sum()), int(((pred == 1) & (truth == 0)).sum())
        from statsmodels.stats.proportion import proportion_confint

        for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp)):
            lo, hi = proportion_confint(k, n, alpha=1 - spec.ci_level, method="wilson")
            ests.append(Estimate(f"{name} at threshold {float(thr_fixed):g}", k / n, (float(lo), float(hi)), spec.ci_level, "Wilson"))
    else:
        j = tpr - fpr
        i = int(np.argmax(j))
        ests.append(Estimate("Youden-optimal threshold", float(thr[i]), note=f"sens {tpr[i]:.3f}, spec {1 - fpr[i]:.3f}; chosen on these data so optimistic"))
        warnings.append("No pre-specified threshold (params.threshold); the Youden point is data-derived and optimistic.")
    return TestResult(
        test="roc_auc",
        n={"positive": n1, "negative": n0},
        statistic_name="z (AUC = 0.5)",
        statistic=float((auc - 0.5) / se) if se > 0 else None,
        p=p,
        estimates=ests,
        effect_size=Estimate("AUC", auc, ci, spec.ci_level, "DeLong"),
        checks=[
            CheckResult("positive_class", "Positive class defined explicitly", "positive_label" in spec.params, detail=f"positive = {spec.params.get('positive_label', 1)!r} ({n1} cases)" + ("" if "positive_label" in spec.params else "; set params.positive_label to be explicit")),
            checks.note("ci_method", "No CI in sklearn. Use a stratified bootstrap or DeLong", "DeLong CI reported"),
        ],
        cross_check=cc,
        warnings=warnings,
        extra={"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": [float(t) if np.isfinite(t) else None for t in thr]}},
    )


@procedure("delong", requires=("outcome", "columns"))
def delong(df: pd.DataFrame, spec: Spec) -> TestResult:
    truth = _truth(df, spec, "outcome")
    cols = spec.variables.columns or []
    if len(cols) != 2:
        raise SpecError("delong needs variables.columns = [score_a, score_b]")
    a = column(df, cols[0], "columns[0]")
    b = column(df, cols[1], "columns[1]")
    keep = ~(np.isnan(a) | np.isnan(b))
    truth, a, b = truth[keep], a[keep], b[keep]
    res = delong_paired(truth, a, b, spec.ci_level)
    # cross-check: paired bootstrap of the AUC difference
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(int(spec.params.get("seed", 0)))
    n_boot = int(spec.params.get("n_boot", 2000))
    diffs = []
    idx1, idx0 = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
    for _ in range(n_boot):  # stratified so every resample has both classes
        i = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        diffs.append(roc_auc_score(truth[i], a[i]) - roc_auc_score(truth[i], b[i]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [(1 - spec.ci_level) / 2 * 100, (1 + spec.ci_level) / 2 * 100])
    boot_se = float(diffs.std(ddof=1))
    cc = compare(
        f"paired stratified bootstrap of the AUC difference ({n_boot} resamples)",
        "A paired bootstrap of the AUC difference should give a similar CI. Use it to validate your DeLong implementation.",
        {"se": res["se"], "ci_low": res["ci"][0], "ci_high": res["ci"][1]},
        {"se": boot_se, "ci_low": float(lo), "ci_high": float(hi)},
        kind="exact",
        rtol=0.3,
        atol=0.02,
        detail="SE and CI bounds within 30% relative or 0.02 absolute count as agreement",
    )
    return TestResult(
        test="delong",
        n={"positive": int(truth.sum()), "negative": int(len(truth) - truth.sum())},
        statistic_name="DeLong z",
        statistic=res["z"],
        p=res["p"],
        estimates=[
            Estimate(f"AUC[{cols[0]}]", res["auc_a"]),
            Estimate(f"AUC[{cols[1]}]", res["auc_b"]),
            Estimate("bootstrap CI of the difference", float(diffs.mean()), (float(lo), float(hi)), spec.ci_level, "percentile"),
        ],
        effect_size=Estimate(f"AUC difference ({cols[0]} − {cols[1]})", res["diff"], (float(res["ci"][0]), float(res["ci"][1])), spec.ci_level, "DeLong"),
        checks=[
            checks.manual("same_patients", "Same patients and the same outcome labels for both scores", f"{len(truth)} paired rows"),
            checks.manual("not_nested", "Not a model compared with the same model plus one marker", "for nested models test the added coefficient instead"),
        ],
        cross_check=cc,
        extra={"delong_se": res["se"], "bootstrap_se": boot_se},
    )


# ---------------------------------------------------------------------------
# Intraclass correlation (judgment: ICC form)
# ---------------------------------------------------------------------------
_ICC_FORMS = {
    "ICC1": "ICC1", "ICC(1,1)": "ICC1", "ICC(1)": "ICC1",
    "ICC2": "ICC2", "ICC(2,1)": "ICC2", "ICC(A,1)": "ICC2",
    "ICC3": "ICC3", "ICC(3,1)": "ICC3", "ICC(C,1)": "ICC3",
    "ICC1K": "ICC1k", "ICC(1,K)": "ICC1k",
    "ICC2K": "ICC2k", "ICC(2,K)": "ICC2k", "ICC(A,K)": "ICC2k",
    "ICC3K": "ICC3k", "ICC(3,K)": "ICC3k", "ICC(C,K)": "ICC3k",
}
_ICC_PG = {"ICC1": "ICC1", "ICC2": "ICC2", "ICC3": "ICC3", "ICC1k": "ICC1k", "ICC2k": "ICC2k", "ICC3k": "ICC3k"}
_ICC_DESC = {
    "ICC1": "ICC(1,1): one-way random, single rater, absolute agreement",
    "ICC2": "ICC(2,1): two-way random, single rater, absolute agreement",
    "ICC3": "ICC(3,1): two-way mixed, single rater, consistency",
    "ICC1k": "ICC(1,k): one-way random, average of k raters",
    "ICC2k": "ICC(2,k): two-way random, average of k raters, absolute agreement",
    "ICC3k": "ICC(3,k): two-way mixed, average of k raters, consistency",
}


@procedure("icc", requires=("columns|outcome",), decisions=("icc_form",))
def icc(df: pd.DataFrame, spec: Spec) -> TestResult:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    form_raw = str(spec.decisions["icc_form"]).upper().replace(" ", "")
    form = _ICC_FORMS.get(form_raw)
    if form is None:
        raise SpecError(f"decisions.icc_form must be one of {sorted(set(_ICC_PG))}; got {spec.decisions['icc_form']!r}")
    m = blocks(df, spec)
    n, k = m.shape
    if k < 2:
        raise SpecError("icc needs at least two raters / measurements")
    long = m.reset_index().melt(id_vars=m.index.name or "index", var_name="rater", value_name="score").rename(columns={m.index.name or "index": "target"})
    ols = smf.ols("score ~ C(target) + C(rater)", long).fit()
    tab = sm.stats.anova_lm(ols)
    msr, msc, mse = (float(tab.loc[i, "mean_sq"]) for i in ("C(target)", "C(rater)", "Residual"))
    ols1 = smf.ols("score ~ C(target)", long).fit()
    tab1 = sm.stats.anova_lm(ols1)
    msr1, msw = float(tab1.loc["C(target)", "mean_sq"]), float(tab1.loc["Residual", "mean_sq"])
    # Shrout & Fleiss (1979) / McGraw & Wong (1996)
    manual = {
        "ICC1": (msr1 - msw) / (msr1 + (k - 1) * msw),
        "ICC2": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
        "ICC3": (msr - mse) / (msr + (k - 1) * mse),
        "ICC1k": (msr1 - msw) / msr1,
        "ICC2k": (msr - mse) / (msr + (msc - mse) / n),
        "ICC3k": (msr - mse) / msr,
    }
    value = float(manual[form])
    ci = None
    cc = None
    try:
        import pingouin as pg

        res = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="score")
        # pingouin labels the rows ICC(1,1), ICC(A,1), ICC(C,1), ICC(1,k), ICC(A,k), ICC(C,k) in a fixed order
        pg_pos = {"ICC1": 0, "ICC2": 1, "ICC3": 2, "ICC1k": 3, "ICC2k": 4, "ICC3k": 5}
        row = res.iloc[pg_pos[form]]
        ci_col = [c for c in res.columns if c.startswith("CI")][0]
        ci_raw = row[ci_col]
        ci = (float(ci_raw[0]), float(ci_raw[1]))
        cc = compare(
            "two-way ANOVA mean squares in the Shrout–Fleiss formula",
            "Plug the mean squares into the Shrout–Fleiss formula. It should reproduce pingouin's row.",
            {"icc": float(row["ICC"])},
            {"icc": value},
            rtol=1e-6,
        )
        value = float(row["ICC"])
        extra_all = {t: float(res.iloc[i]["ICC"]) for t, i in pg_pos.items()}
    except ImportError:
        extra_all = manual
    chk = [
        checks.note("icc_form", "Form chosen: one-way or two-way, random or fixed raters, agreement or consistency, single or average", _ICC_DESC[form] + "; chosen by the user"),
        checks.shapiro(np.asarray(ols.resid), "residual_normality", "Residuals roughly normal", spec.alpha),
    ]
    dropped = m.attrs.get("dropped", 0)
    if dropped:
        chk.append(CheckResult("complete_ratings", "Every target rated by every rater", False, detail=f"{dropped} incomplete target(s) dropped"))
    return TestResult(
        test="icc",
        n={"targets": int(n), "raters": int(k)},
        statistic_name="F (targets)",
        statistic=float(tab.loc["C(target)", "F"]),
        p=float(tab.loc["C(target)", "PR(>F)"]),
        df=(float(tab.loc["C(target)", "df"]), float(tab.loc["Residual", "df"])),
        effect_size=Estimate(_ICC_DESC[form].split(":")[0], value, ci, spec.ci_level if ci else None, _ICC_DESC[form].split(": ")[1]),
        estimates=[Estimate(f"{t} (not chosen)", v) for t, v in extra_all.items() if t != form],
        checks=chk,
        cross_check=cc,
        extra={"form": form, "mean_squares": {"targets": msr, "raters": msc, "residual": mse}},
    )
