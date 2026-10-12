"""Round-three acceptance: public API vs explicit synthetic reference fits.

Only Spec/run are imported from whichtest. References are fitted directly;
no primary expectations are taken from the pipeline's cross-check output.
"""
import copy
from math import log, sqrt

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from lifelines import CoxPHFitter
from statsmodels.multivariate.factor import Factor
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

from whichtest import Spec, SpecError, run


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(9031)
    m, k = 60, 5
    d = pd.DataFrame({"id": np.repeat(np.arange(m), k), "visit": np.tile(np.arange(k), m)})
    d["x"] = np.repeat(rng.normal(size=m), k) + rng.normal(0, .6, len(d))
    d["g"] = np.repeat(np.arange(m) % 2, k)
    u = np.repeat(rng.normal(0, 1, m), k)
    d["y"] = 2 + .5*d.x + .4*d.visit + u + rng.normal(0, .7, len(d))
    d["binary"] = rng.binomial(1, 1/(1+np.exp(-(-.4+.6*d.x+.6*u))))
    d["exposure"] = rng.uniform(.7, 2, len(d))
    d["count"] = rng.poisson(d.exposure*np.exp(.2+.3*d.x+.3*u))
    latent = .6*d.x + .8*u + rng.logistic(size=len(d))
    d["ordinal"] = np.digitize(latent, [-.5, .8])
    logits = np.column_stack([.4+.6*d.x+.5*u, -.3-.4*d.x-.4*u, np.zeros(len(d))])
    prob = np.exp(logits); prob /= prob.sum(axis=1, keepdims=True)
    d["nominal"] = [rng.choice(3, p=p) for p in prob]
    t = np.tile(np.arange(-2., 3.), m)
    b = pd.DataFrame({"id": d.id, "t": t})
    b["y"] = 2 + .4*t + np.repeat(rng.normal(0, 1.5, m), k) + rng.normal(0, .5, len(b))
    slopes = b.copy()
    slopes["y"] += np.repeat(rng.normal(0, .9, m), k)*t
    surv = pd.DataFrame({"site": np.repeat(np.arange(45), 6)})
    frailty = np.repeat(rng.normal(0, 1, 45), 6)
    surv["x"] = np.repeat(rng.normal(size=45), 6) + rng.normal(0, .4, len(surv))
    surv["time"] = rng.exponential(np.exp(-(.15*surv.x+frailty)))
    surv["event"] = rng.binomial(1, .8, len(surv))
    surv["cause"] = np.where(surv.event, rng.choice([1,2], len(surv), p=[.65,.35]), 0)
    surv["stratum"] = np.repeat(np.arange(45) % 3, 6)
    surv["time"] *= np.exp(.9*surv.stratum)
    z = pd.DataFrame({"x": rng.normal(size=700)})
    mu = np.exp(.8+.45*z.x)
    z["y"] = np.where(rng.random(len(z)) < 1/(1+np.exp(-(-.4+.6*z.x))), 0,
                       rng.negative_binomial(2, 2/(2+mu)))
    factors = rng.normal(size=(350,2))
    factors[:,1] = .6*factors[:,0] + .8*factors[:,1]
    load = np.array([[.9,.1],[.8,.2],[.7,0],[.1,.9],[.2,.8],[0,.7]])
    items = pd.DataFrame(factors@load.T+rng.normal(0,.45,(350,6)), columns=[f"i{i}" for i in range(6)])
    cells = []
    for a, c, n in [(0,0,15),(0,1,45),(1,0,55),(1,1,20)]:
        cells.extend({"a": str(a), "b": str(c), "y": .5*a+.4*c+2*a*c+v} for v in rng.normal(size=n))
    ratings = pd.DataFrame(rng.normal(size=(18,3))+.6*np.arange(3)+np.arange(18)[:,None]/3, columns=list("abc"))
    return {"long": d, "balanced": b, "slopes": slopes, "survival": surv,
            "zero": z, "items": items, "factorial": pd.DataFrame(cells), "ratings": ratings}


def checked(test, variables, decisions, data, **kwargs):
    original = copy.deepcopy(decisions)
    result = run(Spec(test=test, variables=variables, decisions=decisions, **kwargs), data)
    assert result.extra["decisions"] == original
    assert decisions == original, "execution must not modify the user's decisions"
    return result


def estimate_map(result):
    return {e.name: e for e in result.estimates}


def assert_coefficients(result, params, ses, prefix="", exp=False, tol=1e-6):
    estimates = estimate_map(result)
    for name, value in params.items():
        e = estimates[prefix+str(name)]
        bounds = np.array([value-1.959963984540054*ses[name], value+1.959963984540054*ses[name]])
        assert e.value == pytest.approx(float(np.exp(value) if exp else value), rel=tol, abs=tol)
        assert e.ci == pytest.approx(np.exp(bounds) if exp else bounds, rel=tol, abs=tol)


def test_lmm_balanced_reml_matches_variance_components(samples):
    d = samples["balanced"]
    y = d.y.to_numpy().reshape(60,5); t = np.arange(-2.,3.)
    beta = (y*t).sum()/(60*(t*t).sum())
    grand, means = y.mean(), y.mean(axis=1)
    sigma = ((y-means[:,None]-beta*t)**2).sum()/(60*4-1)
    tau = (5*((means-grand)**2).sum()/59-sigma)/5
    r = checked("lmm", {"outcome":"y","predictors":["t"],"subject":"id"},
                {"re_formula":"1","reml":True}, d)
    e = estimate_map(r)
    assert e["Intercept"].value == pytest.approx(grand, abs=1e-5)
    assert e["t"].value == pytest.approx(beta, abs=1e-5)
    assert r.extra["cov_re"][0][0] == pytest.approx(tau, rel=3e-3)
    assert r.effect_size.value == pytest.approx(tau/(tau+sigma), rel=3e-3)


def test_lmm_random_structure_and_reml_affect_actual_fit(samples):
    d = samples["slopes"].copy()
    d["y"] += 2*(d.groupby("id").y.transform("mean")-d.y.mean())
    fits = {}
    for re_formula, reml in [("1",True),("~t",True),("~t",False)]:
        ref = smf.mixedlm("y ~ t", d, groups=d.id, re_formula=re_formula).fit(reml=reml, method="powell", disp=False)
        assert ref.converged
        r = checked("lmm", {"outcome":"y","predictors":["t"],"subject":"id"},
                    {"re_formula":re_formula,"reml":reml}, d)
        assert_coefficients(r, ref.fe_params, ref.bse_fe, tol=3e-3)
        np.testing.assert_allclose(r.extra["cov_re"], ref.cov_re, rtol=.01, atol=.003)
        assert r.extra["llf"] == pytest.approx(ref.llf, rel=1e-5)
        fits[re_formula,reml] = r
    assert np.shape(fits["1",True].extra["cov_re"]) == (1,1)
    assert np.shape(fits["~t",True].extra["cov_re"]) == (2,2)
    assert fits["~t",True].extra["cov_re"][1][1] > .1
    assert not np.allclose(fits["~t",True].extra["cov_re"], fits["~t",False].extra["cov_re"])


@pytest.mark.parametrize("tid,outcome,family,prefix", [
    ("gee_cont","y",sm.families.Gaussian,""),
    ("gee_logit","binary",sm.families.Binomial,"OR "),
    ("gee_poisson","count",sm.families.Poisson,"RR "),
])
def test_gee_independence_grouped_closed_form_and_correlation_decision(samples, tid, outcome, family, prefix):
    d = samples["long"]
    v = {"outcome":outcome,"predictors":["g"],"subject":"id"}
    if tid == "gee_poisson": v["exposure"] = "exposure"
    means = [d.loc[d.g==g,outcome].mean() for g in [0,1]]
    if tid == "gee_logit": transformed = [log(p/(1-p)) for p in means]
    elif tid == "gee_poisson": transformed = [log(d.loc[d.g==g,outcome].sum()/d.loc[d.g==g,"exposure"].sum()) for g in [0,1]]
    else: transformed = means
    expected = np.array([transformed[0], transformed[1]-transformed[0]])
    results = []
    for wc, cs in [("independence", sm.cov_struct.Independence()), ("exchangeable", sm.cov_struct.Exchangeable())]:
        offset = np.log(d.exposure) if tid == "gee_poisson" else None
        X = pd.DataFrame({"Intercept":np.ones(len(d)), "g":d.g})
        ref = sm.GEE(d[outcome], X, groups=d.id, family=family(), cov_struct=cs, offset=offset).fit()
        assert ref.converged
        r = checked(tid, v, {"working_correlation":wc}, d)
        assert_coefficients(r, ref.params, ref.bse, prefix, exp=bool(prefix))
        if wc == "independence":
            actual = [estimate_map(r)[prefix+n].value for n in ["Intercept","g"]]
            np.testing.assert_allclose(actual, np.exp(expected) if prefix else expected, rtol=1e-6)
        results.append(r)
    # With equal group sizes and only a cluster-level group indicator, working
    # correlation can algebraically cancel. Use within-cluster x to test changes.
    v["predictors"] = ["x"]
    varied = []
    for wc, cs in [("independence", sm.cov_struct.Independence()), ("exchangeable", sm.cov_struct.Exchangeable())]:
        X = pd.DataFrame({"Intercept":np.ones(len(d)), "x":d.x})
        ref = sm.GEE(d[outcome], X, groups=d.id, family=family(), cov_struct=cs, offset=offset).fit()
        assert ref.converged
        r = checked(tid, v, {"working_correlation":wc}, d)
        assert_coefficients(r, ref.params, ref.bse, prefix, exp=bool(prefix))
        varied.append(estimate_map(r)[prefix+"x"].value)
    assert not np.isclose(*varied, rtol=1e-5, atol=1e-5)


def test_lmm_optimizer_recovers_from_singular_first_attempt(samples):
    d = samples["slopes"]
    ref = smf.mixedlm("y ~ t", d, groups=d.id, re_formula="1").fit(reml=True, method="powell", disp=False)
    assert ref.converged and np.isfinite(ref.bse_fe).all()
    r = checked("lmm", {"outcome":"y","predictors":["t"],"subject":"id"}, {"re_formula":"1","reml":True}, d)
    assert_coefficients(r, ref.fe_params, ref.bse_fe, tol=3e-3)
    np.testing.assert_allclose(r.extra["cov_re"], ref.cov_re, rtol=.01, atol=.003)


@pytest.mark.parametrize("tid", ["ordinal_gee","nominal_gee"])
def test_categorical_gee_direct_fit_and_working_correlation(samples, tid):
    d = samples["long"]
    outcome = "ordinal" if tid == "ordinal_gee" else "nominal"
    estimates = []
    for wc in ["independence","global_odds_ratio"]:
        cs = sm.cov_struct.Independence() if wc=="independence" else sm.cov_struct.GlobalOddsRatio(outcome)
        X = d[["x"]] if tid=="ordinal_gee" else sm.add_constant(d[["x"]])
        cls = sm.OrdinalGEE if tid=="ordinal_gee" else sm.NominalGEE
        ref = cls(d[outcome].to_numpy(), X, groups=d.id.to_numpy(), cov_struct=cs).fit()
        assert ref.converged
        r = checked(tid, {"outcome":outcome,"predictors":["x"],"subject":"id"}, {"working_correlation":wc}, d)
        if tid=="ordinal_gee":
            assert_coefficients(r, ref.params.loc[["x"]], ref.bse.loc[["x"]], "cumulative OR ", True)
        else:
            # Reference names expose category-specific coefficients directly.
            for j in [0,1]:
                name = next(n for n in ref.params.index if n.startswith("x[") and float(n.split("[")[1][:-1])==j)
                e = estimate_map(r)[f"OR[{j} vs 2] x"]
                assert e.value == pytest.approx(np.exp(ref.params[name]), rel=1e-6)
                assert e.ci == pytest.approx(np.exp(ref.params[name]+np.array([-1,1])*stats.norm.ppf(.975)*ref.bse[name]), rel=1e-6)
        estimates.append([e.value for e in r.estimates])
    assert not np.allclose(estimates[0], estimates[1], rtol=1e-5, atol=1e-5)


def test_factorial_ss_type_and_interaction_decisions(samples):
    d = samples["factorial"]
    results = {}
    for ss, interaction in [(2,True),(3,True),(2,False)]:
        f = "y ~ "+(" * " if interaction else " + ").join(f"C({a}, Sum)" if ss==3 else f"C({a})" for a in ["a","b"])
        table = sm.stats.anova_lm(smf.ols(f,d).fit(), typ=ss)
        r = checked("factorial_anova", {"outcome":"y","predictors":["a","b"]}, {"ss_type":ss,"include_interaction":interaction},d)
        for name,e in estimate_map(r).items():
            term = name.removesuffix(": F")
            refterm = term.replace("C(a)","C(a, Sum)").replace("C(b)","C(b, Sum)") if ss==3 else term
            assert e.value == pytest.approx(table.loc[refterm,"F"], rel=1e-7)
        results[ss,interaction] = r
    assert not np.isclose(estimate_map(results[2,True])["C(a): F"].value, estimate_map(results[3,True])["C(a): F"].value)
    assert "C(a):C(b): F" not in estimate_map(results[2,False])
    assert not np.isclose(results[2,False].statistic, results[2,True].statistic)


def test_zero_inflated_distribution_and_inflation_formula_decisions(samples):
    d=samples["zero"]; X=pd.DataFrame({"Intercept":np.ones(len(d)),"x":d.x})
    results={}
    for dist,infl in [("poisson","1"),("poisson","x"),("negbin","x")]:
        Xi = X if infl=="x" else X[["Intercept"]]
        cls = ZeroInflatedPoisson if dist=="poisson" else ZeroInflatedNegativeBinomialP
        ref=cls(d.y, X, exog_infl=Xi).fit(method="bfgs",maxiter=3000,disp=0)
        assert ref.mle_retvals["converged"] and np.isfinite(ref.bse).all()
        r=checked("zip",{"outcome":"y","predictors":["x"]},{"count_distribution":dist,"inflation_formula":infl},d)
        es=estimate_map(r)
        for name,b in ref.params.items():
            key = "zero part OR "+name[8:] if name.startswith("inflate_") else "NB dispersion α" if name=="alpha" else "count part RR "+name
            assert es[key].value == pytest.approx(b if name=="alpha" else np.exp(b), rel=.003, abs=.0005)
        assert r.extra["aic"]["zero_inflated"] == pytest.approx(ref.aic, rel=1e-5)
        results[dist,infl]=r
    assert "zero part OR x" not in estimate_map(results["poisson","1"])
    assert "zero part OR x" in estimate_map(results["poisson","x"])
    assert "NB dispersion α" in estimate_map(results["negbin","x"])
    assert abs(results["poisson","x"].extra["aic"]["zero_inflated"]-results["negbin","x"].extra["aic"]["zero_inflated"]) > 2


def test_competing_both_estimands_and_event_choice(samples):
    # CIF reference from hand risk sets, unique event times, no jitter.
    hand=pd.DataFrame({"time":[1,2,3,4,5,6],"cause":[1,2,0,1,0,0]})
    for cause,expected in [(1,7/18),(2,1/6)]:
        r=checked("competing",{"time":"time","event":"cause"},{"estimand":"cumulative_incidence","event_of_interest":cause},hand,params={"timepoints":[4]})
        assert estimate_map(r)[f"cumulative incidence of {cause} at 4"].value == pytest.approx(expected)
        assert r.statistic is None and r.p is None
    d=samples["survival"]
    r=checked("competing",{"time":"time","event":"cause","predictors":["x"]},{"estimand":"cause_specific_hazard","event_of_interest":1},d)
    ref=CoxPHFitter().fit(d[["time","x"]].assign(event=(d.cause==1).astype(int)),"time","event")
    assert_coefficients(r,ref.params_,ref.standard_errors_,"cause-specific HR ",True,tol=1e-4)
    assert r.statistic == pytest.approx(ref.log_likelihood_ratio_test().test_statistic,rel=1e-5)
    assert r.p == pytest.approx(ref.log_likelihood_ratio_test().p_value,rel=1e-4)


@pytest.mark.parametrize("approach",["cluster_robust","stratify"])
def test_cox_approach_matches_lifelines_coefficients_and_intervals(samples,approach):
    d=samples["survival"]
    v={"time":"time","event":"event","predictors":["x"],"cluster":"site","strata":"stratum"}
    r=checked("cox_cluster",v,{"approach":approach},d)
    col="site" if approach=="cluster_robust" else "stratum"
    ref=CoxPHFitter().fit(d[["time","event","x",col]],"time","event",**({"cluster_col":col} if approach=="cluster_robust" else {"strata":[col]}))
    assert_coefficients(r,ref.params_,ref.standard_errors_,"HR ",True,tol=2e-4)
    if approach=="stratify":
        assert r.statistic == pytest.approx(ref.log_likelihood_ratio_test().test_statistic,rel=1e-5)
    else:
        # The robust approach must produce cluster-aware headline inference.
        robust_p=float(ref.summary.loc["x","p"])
        assert r.p == pytest.approx(robust_p,rel=1e-4,abs=1e-10), f"headline p={r.p}; robust single-coefficient Wald p={robust_p}"


def test_cox_approach_changes_computation(samples):
    d=samples["survival"]; v={"time":"time","event":"event","predictors":["x"],"cluster":"site","strata":"stratum"}
    results=[checked("cox_cluster",v,{"approach":a},d) for a in ["cluster_robust","stratify"]]
    assert not np.isclose(results[0].estimates[0].value,results[1].estimates[0].value,rtol=1e-4)


def test_exploratory_factor_choices_match_direct_factor_fit(samples):
    d=samples["items"]; results={}
    for k,rotation in [(1,"none"),(2,"none"),(2,"varimax")]:
        corr = np.corrcoef(d, rowvar=False)
        values, vectors = np.linalg.eigh(corr)
        start_load = vectors[:,-k:] * np.sqrt(np.maximum(values[-k:]-.5,.05))
        start_unique = np.maximum(1-(start_load**2).sum(axis=1),.1)
        ref=Factor(d,n_factor=k,method="ml").fit(start=(start_load,start_unique))
        if rotation!="none": ref.rotate(rotation)
        r=checked("factor_sem",{"columns":list(d.columns)},{"approach":"exploratory","n_factors":k,"rotation":rotation},d)
        L=np.array([r.extra["loadings"][c] for c in d.columns])
        # Covariance removes arbitrary factor signs and permutations.
        R=np.real(ref.loadings)
        np.testing.assert_allclose(L@L.T,R@R.T,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose([r.extra["uniqueness"][c] for c in d.columns],np.real(ref.uniqueness),rtol=1e-5)
        results[k,rotation]=L
    assert results[1,"none"].shape==(6,1) and results[2,"none"].shape==(6,2)
    assert not np.allclose(np.abs(results[2,"none"]),np.abs(results[2,"varimax"]),atol=.01)


def test_icc_decision_forms_from_mean_squares(samples):
    d=samples["ratings"]; a=d.to_numpy(); n,k=a.shape
    row,col,grand=a.mean(1),a.mean(0),a.mean()
    msr=k*np.sum((row-grand)**2)/(n-1); msc=n*np.sum((col-grand)**2)/(k-1)
    mse=np.sum((a-row[:,None]-col+grand)**2)/((n-1)*(k-1)); msw=np.sum((a-row[:,None])**2)/(n*(k-1))
    vals={"ICC1":(msr-msw)/(msr+(k-1)*msw),"ICC2":(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n),"ICC3":(msr-mse)/(msr+(k-1)*mse)}
    for form in list(vals): vals[form+"k"]=k*vals[form]/(1+(k-1)*vals[form])
    for form,value in vals.items():
        r=checked("icc",{"columns":list(d.columns)},{"icc_form":form},d)
        assert r.effect_size.value==pytest.approx(value)
        assert r.statistic==pytest.approx(msr/(msw if form in ["ICC1","ICC1k"] else mse))
    assert len(set(round(v,6) for v in vals.values()))==6


# Required keys come from the documented public contract, not the registry.
CASES={
 "factorial_anova":("factorial",{"outcome":"y","predictors":["a","b"]},{"ss_type":2,"include_interaction":False}),
 "lmm":("balanced",{"outcome":"y","predictors":["t"],"subject":"id"},{"re_formula":"1","reml":True}),
 "gee_cont":("long",{"outcome":"y","predictors":["x"],"subject":"id"},{"working_correlation":"independence"}),
 "gee_logit":("long",{"outcome":"binary","predictors":["x"],"subject":"id"},{"working_correlation":"independence"}),
 "gee_poisson":("long",{"outcome":"count","predictors":["x"],"subject":"id","exposure":"exposure"},{"working_correlation":"independence"}),
 "ordinal_gee":("long",{"outcome":"ordinal","predictors":["x"],"subject":"id"},{"working_correlation":"independence"}),
 "nominal_gee":("long",{"outcome":"nominal","predictors":["x"],"subject":"id"},{"working_correlation":"independence"}),
 "zip":("zero",{"outcome":"y","predictors":["x"]},{"count_distribution":"poisson","inflation_formula":"1"}),
 "competing":("survival",{"time":"time","event":"cause","predictors":["x"]},{"estimand":"cumulative_incidence","event_of_interest":1}),
 "cox_cluster":("survival",{"time":"time","event":"event","predictors":["x"],"cluster":"site"},{"approach":"cluster_robust"}),
 "factor_sem":("items",{"columns":[f"i{i}" for i in range(6)]},{"approach":"exploratory","n_factors":2,"rotation":"varimax"}),
 "icc":("ratings",{"columns":list("abc")},{"icc_form":"ICC2"}),
}
MISSING=[(tid,key) for tid,(_,_,dec) in CASES.items() for key in dec]


@pytest.mark.parametrize("tid,key",MISSING,ids=[f"{t}-{k}" for t,k in MISSING])
def test_judgment_missing_each_required_key_is_rejected(samples,tid,key):
    data,v,dec=CASES[tid]; dec=dict(dec); del dec[key]
    error = None
    try:
        run(Spec(test=tid,variables=v,decisions=dec),samples[data])
    except SpecError:
        return
    except Exception as exc:
        error = type(exc).__name__
    pytest.fail(f"expected SpecError; got {error or 'successful execution'}")


INVALID=[
 ("factorial_anova","ss_type",2.9),("factorial_anova","ss_type",4),
 ("factorial_anova","include_interaction","false"),
 ("lmm","reml","false"),("lmm","re_formula","~missing_random_column"),
 *[(tid,"working_correlation","invalid") for tid in ["gee_cont","gee_logit","gee_poisson","ordinal_gee","nominal_gee"]],
 ("zip","count_distribution","invalid"),("zip","inflation_formula","missing_inflation_column"),
 ("competing","estimand","invalid"),("competing","event_of_interest",0),
 ("cox_cluster","approach","invalid"),("factor_sem","approach","invalid"),
 ("factor_sem","n_factors",1.5),("factor_sem","rotation","invalid"),("icc","icc_form","invalid"),
]


@pytest.mark.parametrize("tid,key,value",INVALID,ids=[f"{t}-{k}-{v}" for t,k,v in INVALID])
def test_judgment_invalid_decision_is_spec_error(samples,tid,key,value):
    data,v,dec=CASES[tid]; dec={**dec,key:value}
    error = None
    try:
        run(Spec(test=tid,variables=v,decisions=dec),samples[data])
    except SpecError:
        return
    except Exception as exc:
        error = type(exc).__name__
    pytest.fail(f"expected SpecError; got {error or 'successful execution'}")
