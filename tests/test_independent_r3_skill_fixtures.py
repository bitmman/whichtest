"""Runnable golden specs plus descriptions/rubrics for a later model evaluation.

These tests validate the target specs. They do not claim to evaluate a model's
ability to follow the natural-language skill.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from whichtest import Spec, run

CASES=json.loads((Path(__file__).parent/"fixtures/r3_study_cases.json").read_text())


@pytest.mark.parametrize("case",CASES,ids=lambda c:c["id"])
def test_study_golden_spec_resolves_and_runs_against_reference(case):
    rng=np.random.default_rng(31204)
    s=Spec(**case["expected_spec"])
    assert Spec.from_design(s.design).test==s.test
    assert s.complete().test==s.test
    if case["id"]=="paired_change":
        d=pd.DataFrame({"sbp_before":rng.normal(140,10,30)})
        d["sbp_after"]=d.sbp_before+rng.normal(-4,5,30)
        delta=d.sbp_after-d.sbp_before
        expected=delta.mean()/(delta.std(ddof=1)/np.sqrt(30))
        r=run(s,d)
        assert r.statistic==pytest.approx(expected)
        assert r.p==pytest.approx(2*stats.t.sf(abs(expected),29))
    elif case["id"]=="adjusted_parallel_trial":
        d=pd.DataFrame({"arm":np.repeat(["control","drug"],100),"sbp_baseline":rng.normal(140,10,200),"age":rng.normal(60,8,200)})
        d["sbp_week12"]=30+.7*d.sbp_baseline+.1*d.age-5*(d.arm=="drug")+rng.normal(0,5,200)
        ref=smf.ols("sbp_week12 ~ C(arm) + sbp_baseline + age",d).fit()
        r=run(s,d)
        assert r.statistic==pytest.approx(ref.fvalue)
        for e in r.estimates:
            assert e.value==pytest.approx(ref.params[e.name])
    else:
        m=70
        d=pd.DataFrame({"id":np.repeat(np.arange(m),4),"week":np.tile([0.,4.,8.,12.],m),"arm":np.repeat(np.where(np.arange(m)%2,"drug","control"),4)})
        d["sbp"]=140-.2*d.week-.3*d.week*(d.arm=="drug")+np.repeat(rng.normal(0,5,m),4)+np.repeat(rng.normal(0,.4,m),4)*d.week+rng.normal(0,2,len(d))
        d=d.loc[~((d.id%7==0)&(d.week==8))].copy()
        ref=smf.mixedlm("sbp ~ C(arm) * week",d,groups=d.id,re_formula="~week").fit(reml=True,method="powell",disp=False)
        assert ref.converged
        r=run(s,d)
        es={e.name:e for e in r.estimates}
        for key,b in ref.fe_params.items():
            assert es[key].value==pytest.approx(b,abs=.003,rel=.003)
        assert r.extra["decisions"]==s.decisions
