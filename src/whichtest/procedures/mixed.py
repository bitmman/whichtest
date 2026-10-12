"""Clustered / longitudinal models: linear mixed model and the GEE family.

All of these are *judgment* tests: they refuse to run until ``spec.decisions``
carries the human choices the taxonomy names (random-effects structure,
REML, working correlation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .. import checks
from ..registry import procedure
from ..result import CheckResult, Estimate, TestResult, compare
from ..spec import Spec, SpecError
from ..data import formula_columns
from .regression import _formula

_CORR = {
    "independence": sm.cov_struct.Independence,
    "exchangeable": sm.cov_struct.Exchangeable,
    "ar1": lambda: sm.cov_struct.Autoregressive(grid=True),
    "unstructured": sm.cov_struct.Unstructured,
}


def _cov_struct(name: str):
    key = str(name).lower().replace("(", "").replace(")", "").replace("-", "").replace("_", "")
    key = {"ar": "ar1", "autoregressive": "ar1", "exch": "exchangeable", "ind": "independence", "unstr": "unstructured"}.get(key, key)
    if key not in _CORR:
        raise SpecError(f"decisions.working_correlation must be one of {list(_CORR)}; got {name!r}")
    return key, _CORR[key]()


def _cluster_checks(groups: pd.Series, min_clusters: int = 40) -> CheckResult:
    sizes = groups.value_counts()
    return CheckResult(
        "cluster_count",
        f"Enough clusters for robust standard errors (≥ {min_clusters})",
        bool(len(sizes) >= min_clusters),
        float(len(sizes)),
        None,
        f"{len(sizes)} clusters; size min/median/max = {sizes.min()}/{int(sizes.median())}/{sizes.max()}",
    )


def _coef_est(params, bse, names, level, exp=False, prefix="") -> list[Estimate]:
    from scipy import stats

    z = stats.norm.ppf(0.5 + level / 2)
    out = []
    for nm, b, s in zip(names, params, bse):
        lo, hi = b - z * s, b + z * s
        p = float(2 * stats.norm.sf(abs(b / s))) if s > 0 else float("nan")
        if exp:
            b, lo, hi = np.exp(b), np.exp(lo), np.exp(hi)
        out.append(Estimate(f"{prefix}{nm}", float(b), (float(lo), float(hi)), level, f"p = {p:.3g}"))
    return out


# ---------------------------------------------------------------------------
@procedure("lmm", requires=("outcome|formula", "subject"), decisions=("re_formula", "reml"))
def lmm(df: pd.DataFrame, spec: Spec) -> TestResult:
    f = _formula(spec, df)
    subj = spec.variables.subject
    re_formula = str(spec.decisions["re_formula"])  # "1" = random intercept only; "~time" = + random slope
    if not re_formula.startswith("~"):
        re_formula = "~" + re_formula
    reml = bool(spec.decisions["reml"])
    d = df.dropna(subset=formula_columns(f, df) + [subj]).copy()
    model = smf.mixedlm(f, d, groups=d[subj], re_formula=re_formula)
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        res = model.fit(reml=reml, method=["lbfgs", "powell", "cg"])  # try several optimizers before giving up
    boundary = [str(w.message)[:120] for w in caught if "converge" in str(w.message).lower() or "boundary" in str(w.message).lower() or "singular" in str(w.message).lower()]
    fe_names = list(res.fe_params.index)
    # cross-check: GEE with an exchangeable working correlation
    gee = smf.gee(f, subj, d, cov_struct=sm.cov_struct.Exchangeable()).fit()
    cc = compare(
        "GEE (Gaussian, exchangeable) fixed effects",
        "With an identity link, mixed-model fixed effects and GEE estimates target similar quantities. Large disagreement is a warning about the model, not rounding.",
        {f"b[{nm}]": float(res.fe_params[nm]) for nm in fe_names},
        {f"b[{nm}]": float(gee.params[nm]) for nm in fe_names},
        kind="exact",
        rtol=0.25,
        atol=0.1,
        detail="fixed effects within 25% relative or 0.1 absolute count as agreement",
    )
    var_re = float(np.asarray(res.cov_re)[0, 0]) if res.cov_re.size else 0.0
    icc = var_re / (var_re + float(res.scale)) if var_re + res.scale > 0 else None
    chk = [
        checks.note("random_effects", "Random intercepts and slopes chosen from the design", f"re_formula={re_formula!r}, reml={reml}; chosen by the user"),
        CheckResult("convergence", "Model converged without boundary warnings", bool(res.converged and not boundary), None, None, "; ".join(boundary) if boundary else "converged"),
        checks.shapiro(np.asarray(res.resid), "residual_normality", "Residuals roughly normal", spec.alpha),
        _cluster_checks(d[subj], 20),
    ]
    warnings = []
    if chk[1].passed is False:
        warnings.append("Convergence or boundary problems: simplify the random-effects structure or fall back to GEE (gee_cont).")
    return TestResult(
        test="lmm",
        n={"observations": int(res.nobs), "clusters": int(d[subj].nunique())},
        statistic_name="",
        statistic=None,
        p=None,
        estimates=_coef_est(res.fe_params.values, res.bse_fe.values, fe_names, spec.ci_level),
        effect_size=Estimate("ICC (random intercept)", icc, note=f"var(intercept) = {var_re:.4g}, residual var = {float(res.scale):.4g}"),
        checks=chk,
        cross_check=cc,
        warnings=warnings,
        extra={"formula": f, "re_formula": re_formula, "reml": reml, "cov_re": np.asarray(res.cov_re).tolist(), "llf": float(res.llf), "gee_bse": {nm: float(gee.bse[nm]) for nm in fe_names}},
    )


# ---------------------------------------------------------------------------
def _time_positions(d: pd.DataFrame, spec: Spec, name: str) -> np.ndarray | None:
    """Integer visit positions for working correlations that need them (unstructured, AR(1))."""
    tcol = spec.variables.time
    if tcol is None:
        if name == "unstructured":
            raise SpecError("working_correlation='unstructured' needs variables.time (visit position within subject)")
        return None
    tv = pd.to_numeric(d[tcol], errors="coerce")
    if tv.isna().any():
        raise SpecError(f"variables.time={tcol!r} must be numeric visit positions")
    if (tv != np.round(tv)).any():
        raise SpecError(f"variables.time={tcol!r} must hold integer visit positions (0, 1, 2, ...)")
    return tv.to_numpy(int)


def _gee(df: pd.DataFrame, spec: Spec, family, test: str, exp: bool, prefix: str, offset_col: str | None = None) -> TestResult:
    f = _formula(spec, df)
    subj = spec.variables.subject
    name, cs = _cov_struct(spec.decisions["working_correlation"])
    d = df.copy()
    sort_by = [subj] + ([spec.variables.time] if spec.variables.time else [])
    d = d.sort_values(sort_by, kind="stable").reset_index(drop=True)  # AR(1) uses row order within subject
    kw = {}
    if offset_col:
        kw["offset"] = np.log(d[offset_col].to_numpy(float))
    tpos = _time_positions(d, spec, name)
    if tpos is not None:
        kw["time"] = tpos
    res = smf.gee(f, subj, d, family=family, cov_struct=cs, **kw).fit()
    alt_name = "independence" if name != "independence" else "exchangeable"
    alt = smf.gee(f, subj, d, family=family, cov_struct=_CORR[alt_name](), **kw).fit()
    names = list(res.params.index)
    cc = compare(
        f"GEE with {alt_name} working correlation",
        "Robust standard errors should be similar across working correlations. Big swings mean the estimate is fragile.",
        {**{f"b[{nm}]": float(res.params[nm]) for nm in names}, **{f"se[{nm}]": float(res.bse[nm]) for nm in names}},
        {**{f"b[{nm}]": float(alt.params[nm]) for nm in names}, **{f"se[{nm}]": float(alt.bse[nm]) for nm in names}},
        kind="exact",
        rtol=0.25,
        atol=0.05,
        detail="coefficients and robust SEs within 25% relative or 0.05 absolute count as agreement",
    )
    qic = res.qic(scale=res.scale)  # a fixed scale makes QIC comparable across working correlations
    order_note = f"; within-subject order = {spec.variables.time}" if spec.variables.time else ("; within-subject order = row order (set variables.time to be explicit)" if name == "ar1" else "")
    chk = [_cluster_checks(d[subj]), checks.note("working_correlation", "Working correlation chosen", (f"{name}; {cs.summary()}" if hasattr(cs, "summary") else name) + order_note)]
    return TestResult(
        test=test,
        n={"observations": int(res.nobs), "clusters": int(d[subj].nunique())},
        statistic_name="",
        statistic=None,
        p=None,
        estimates=_coef_est(res.params.values, res.bse.values, names, spec.ci_level, exp=exp, prefix=prefix),
        checks=chk,
        cross_check=cc,
        extra={"formula": f, "working_correlation": name, "qic": float(qic[0]), "qicu": float(qic[1]), "alt_qic": float(alt.qic(scale=res.scale)[0]), "scale": float(res.scale)},
    )


@procedure("gee_cont", requires=("outcome|formula", "subject"), decisions=("working_correlation",))
def gee_cont(df: pd.DataFrame, spec: Spec) -> TestResult:
    r = _gee(df, spec, sm.families.Gaussian(), "gee_cont", False, "")
    r.checks.append(checks.manual("missingness", "Missing data plausibly completely at random (GEE needs MCAR; a mixed model tolerates MAR)"))
    return r


@procedure("gee_logit", requires=("outcome|formula", "subject"), decisions=("working_correlation",))
def gee_logit(df: pd.DataFrame, spec: Spec) -> TestResult:
    r = _gee(df, spec, sm.families.Binomial(), "gee_logit", True, "OR ")
    r.checks.append(checks.manual("estimand", "Population-averaged rather than subject-specific effects are what you want"))
    return r


@procedure("gee_poisson", requires=("outcome|formula", "subject"), decisions=("working_correlation",))
def gee_poisson(df: pd.DataFrame, spec: Spec) -> TestResult:
    ex = spec.variables.exposure
    r = _gee(df, spec, sm.families.Poisson(), "gee_poisson", True, "RR ", offset_col=ex)
    r.checks.insert(0, CheckResult("exposure_offset", "Exposure time included as an offset", ex is not None, detail=f"offset = log({ex})" if ex else "no variables.exposure: rates assume equal exposure"))
    return r


# ---------------------------------------------------------------------------
def _design(spec: Spec, df: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    import patsy

    f = _formula(spec, df)
    yname = spec.variables.outcome or f.split("~")[0].strip()
    d = df.dropna(subset=[yname, spec.variables.subject]).copy()
    X = patsy.dmatrix(f.split("~", 1)[1], d, return_type="dataframe").drop(columns="Intercept", errors="ignore")
    return yname, d.loc[X.index], X


@procedure("ordinal_gee", requires=("outcome|formula", "subject"), decisions=("working_correlation",))
def ordinal_gee(df: pd.DataFrame, spec: Spec) -> TestResult:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    yname, d, X = _design(spec, df)
    y = d[yname].astype(int).to_numpy()
    wc = str(spec.decisions["working_correlation"]).lower()
    if wc in ("global_odds_ratio", "globaloddsratio", "gor"):
        cs, cname = sm.cov_struct.GlobalOddsRatio("ordinal"), "global odds ratio"
    elif wc == "independence":
        cs, cname = sm.cov_struct.Independence(), "independence"
    else:
        raise SpecError("decisions.working_correlation for ordinal_gee must be 'global_odds_ratio' or 'independence'")
    # the formula interface adds an intercept that collides with the thresholds; use the array interface
    res = sm.OrdinalGEE(y, X, groups=d[spec.variables.subject].to_numpy(), cov_struct=cs).fit()
    slope_names = list(X.columns)
    naive = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=0, maxiter=1000)
    cc = compare(
        "OrderedModel (naive, ignoring clustering)",
        "Point estimates should agree in direction. The naive standard errors will be too small.",
        {f"direction[{nm}]": float(res.params[nm]) for nm in slope_names},
        {f"direction[{nm}]": float(naive.params[nm]) for nm in slope_names},
        kind="direction",
        detail="; ".join(f"{nm}: robust SE {float(res.bse[nm]):.3f} vs naive {float(naive.bse[nm]):.3f}" for nm in slope_names),
    )
    counts = pd.Series(y).value_counts().sort_index()
    chk = [
        checks.manual("proportional_odds", "Proportional odds plausible, as for ordinal logistic regression", "run ordinal_logit on the same data for the cut-point check"),
        _cluster_checks(d[spec.variables.subject]),
        CheckResult("sparse_levels", "Every outcome level reasonably populated", bool(counts.min() >= 10), float(counts.min()), None, ", ".join(f"{k}: {v}" for k, v in counts.items())),
    ]
    return TestResult(
        test="ordinal_gee",
        n={"observations": int(len(y)), "clusters": int(d[spec.variables.subject].nunique())},
        statistic_name="",
        statistic=None,
        p=None,
        estimates=_coef_est([float(res.params[nm]) for nm in slope_names], [float(res.bse[nm]) for nm in slope_names], slope_names, spec.ci_level, exp=True, prefix="cumulative OR "),
        checks=chk,
        cross_check=cc,
        extra={"working_correlation": cname, "thresholds": {nm: float(v) for nm, v in res.params.items() if nm not in slope_names}},
    )


@procedure("nominal_gee", requires=("outcome|formula", "subject"), decisions=("working_correlation",))
def nominal_gee(df: pd.DataFrame, spec: Spec) -> TestResult:
    yname, d, X = _design(spec, df)
    cats = pd.Categorical(d[yname])
    y = cats.codes.astype(int)
    wc = str(spec.decisions["working_correlation"]).lower()
    if wc in ("global_odds_ratio", "globaloddsratio", "gor"):
        cs, cname = sm.cov_struct.GlobalOddsRatio("nominal"), "global odds ratio"
    elif wc == "independence":
        cs, cname = sm.cov_struct.Independence(), "independence"
    else:
        raise SpecError("decisions.working_correlation for nominal_gee must be 'global_odds_ratio' or 'independence'")
    Xi = sm.add_constant(X, has_constant="add")
    res = sm.NominalGEE(y, Xi, groups=d[spec.variables.subject].to_numpy(), cov_struct=cs).fit()
    dm = d.assign(**{yname: y})
    naive = smf.mnlogit(f"{yname} ~ {_formula(spec, df).split('~', 1)[1]}", dm).fit(disp=0, maxiter=500)
    # statsmodels NominalGEE compares each category with the *last*; mnlogit with the *first*. Compare directions
    # on the contrast "category j vs reference" after re-basing mnlogit to the last category.
    K = len(cats.categories)
    n_x = Xi.shape[1]
    gee_b = np.asarray(res.params).reshape(K - 1, n_x)  # rows: category j (0..K-2) vs last
    mn = naive.params.to_numpy()  # shape n_x × (K-1): columns category 1..K-1 vs first
    mn_full = np.column_stack([np.zeros(n_x), mn])  # add reference column
    mn_vs_last = (mn_full - mn_full[:, [-1]])[:, :-1].T  # category j vs last
    primary = {f"direction[{cats.categories[j]} vs {cats.categories[-1]}: {nm}]": float(gee_b[j, i]) for j in range(K - 1) for i, nm in enumerate(Xi.columns) if nm != "const"}
    secondary = {f"direction[{cats.categories[j]} vs {cats.categories[-1]}: {nm}]": float(mn_vs_last[j, i]) for j in range(K - 1) for i, nm in enumerate(Xi.columns) if nm != "const"}
    cc = compare(
        "statsmodels mnlogit (naive, ignoring clustering)",
        "The naive multinomial fit ignores clustering. Directions should agree; its standard errors will be too small.",
        primary,
        secondary,
        kind="direction",
    )
    bse = np.asarray(res.bse).reshape(K - 1, n_x)
    ests = []
    for j in range(K - 1):
        for i, nm in enumerate(Xi.columns):
            if nm == "const":
                continue
            ests += _coef_est([gee_b[j, i]], [bse[j, i]], [nm], spec.ci_level, exp=True, prefix=f"OR[{cats.categories[j]} vs {cats.categories[-1]}] ")
    counts = pd.Series(cats).value_counts()
    chk = [
        _cluster_checks(d[spec.variables.subject]),
        CheckResult("sparse_levels", "Every category reasonably populated", bool(counts.min() >= 10), float(counts.min()), None, ", ".join(f"{k}: {v}" for k, v in counts.items())),
    ]
    return TestResult(
        test="nominal_gee",
        n={"observations": int(len(y)), "clusters": int(d[spec.variables.subject].nunique())},
        statistic_name="",
        statistic=None,
        p=None,
        estimates=ests,
        checks=chk,
        cross_check=cc,
        extra={"working_correlation": cname, "reference_category": str(cats.categories[-1])},
    )
