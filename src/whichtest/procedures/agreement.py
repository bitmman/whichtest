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
