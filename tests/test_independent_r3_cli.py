"""Round-three CLI acceptance; shipped trial.csv is documented synthetic data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from whichtest import Spec, SpecError, run

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("spec_*.json"))


def cli(*args):
    return subprocess.run([sys.executable, "-m", "whichtest.cli", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=90,
                          env={**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})


def fmt(x):
    if x is None: return "NA"
    if float(x).is_integer() and abs(x)<1e6: return str(int(x))
    return format(x,".4g")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p:p.stem)
def test_every_example_json_text_strict_and_independent_statistic(path):
    spec=json.loads(path.read_text())
    data=[] if "table" in spec.get("params",{}) else [ROOT/"examples/trial.csv"]
    outputs={}
    for mode in ["json","text"]:
        for strict in [False,True]:
            outputs[mode,strict]=cli("run",path,*data,*( ["--json"] if mode=="json" else []),*( ["--strict"] if strict else []))
    r=json.loads(outputs["json",False].stdout)
    summary=outputs["text",False].stdout
    # Verify status against the actual JSON contract, then require --strict to
    # affect exit status only, for both renderers.
    flagged = (r["cross_check"] is None or not r["cross_check"]["agrees"]
               or any(c["passed"] is False for c in r["checks"]))
    assert r["flagged"] == flagged
    for mode in ["json","text"]:
        assert outputs[mode,False].returncode==0, outputs[mode,False].stderr
        assert outputs[mode,True].returncode==(2 if flagged else 0), outputs[mode,True].stderr
        assert outputs[mode,False].stdout==outputs[mode,True].stdout
    assert f'{r["name"]} [{r["test"]}, {r["status"]}]' in summary
    assert f'{r["statistic_name"]} = {fmt(r["statistic"])}' in summary
    assert "p = "+("<0.0001" if r["p"]<1e-4 else f'{r["p"]:.4f}') in summary
    assert ("FLAGGED" in summary)==flagged
    assert ("DISAGREES" in summary)==(not r["cross_check"]["agrees"])
    for estimate in r["estimates"] + ([r["effect_size"]] if r["effect_size"] else []):
        assert estimate["name"]+" = "+fmt(estimate["value"]) in summary
        if estimate["ci"] is not None:
            level = round(100*(estimate["ci_level"] or .95))
            low, high = estimate["ci"]
            assert f"[{level}% CI {fmt(low)}, {fmt(high)}]" in summary
    for check in r["checks"]:
        marker = {True:"ok  ", False:"FAIL", None:"?   "}[check["passed"]]
        assert f"[{marker}] {check['key']}: {check['text']}" in summary
    for key,value in r.get("extra",{}).get("decisions",{}).items():
        assert f"{key}={value!r}" in summary
    # Primary results independently recomputed from synthetic examples.
    if spec["test"]=="mcnemar":
        table=spec["params"]["table"]; b,c=table[0][1],table[1][0]
        expected_stat=min(b,c); expected_p=stats.binomtest(b,b+c,.5).pvalue
    else:
        d=pd.read_csv(data[0])
        if spec["test"]=="multiple_reg":
            ref=smf.ols("sbp_week12 ~ C(arm) + sbp_baseline + age",d).fit()
            expected_stat,expected_p=ref.fvalue,ref.f_pvalue
        elif spec["test"]=="factorial_anova":
            table=sm.stats.anova_lm(smf.ols("sbp_week12 ~ C(arm, Sum)*C(sex, Sum)",d).fit(),typ=3)
            expected_stat,expected_p=table.loc["C(arm, Sum):C(sex, Sum)",["F","PR(>F)"]]
        else:
            table=pd.crosstab(d.arm,d.responder).to_numpy()
            expected_stat,expected_p,*_=stats.chi2_contingency(table,correction=False)
    assert r["statistic"]==pytest.approx(expected_stat,rel=1e-7)
    assert r["p"]==pytest.approx(expected_p,rel=1e-7)


DESIGN={"goal":"single_outcome","outcome_type":"continuous","structure":"independent","predictors":"one_factor_2_levels"}


@pytest.mark.parametrize("tid,allowed",[("paired_t",False),("mann_whitney",True)])
def test_cli_design_test_consistency_and_fallback(tmp_path,tid,allowed):
    rng=np.random.default_rng(230)
    d=pd.DataFrame({"y":rng.normal(size=80),"arm":np.repeat([0,1],40),"before":rng.normal(size=80),"after":rng.normal(size=80)})
    spec={"test":tid,"design":DESIGN,"variables":{"outcome":"y","group":"arm","before":"before","after":"after"},"reasoning":["Explicit fallback for skewed outcomes"]}
    p=tmp_path/"spec.json"; p.write_text(json.dumps(spec)); f=tmp_path/"data.csv"; d.to_csv(f,index=False)
    completed=cli("run",p,f,"--json")
    if allowed:
        assert completed.returncode==0,completed.stderr
        assert json.loads(completed.stdout)["test"]==tid
        assert Spec(**spec).complete().resolved_test=="indep_t"
    else:
        assert completed.returncode!=0 and "fallback chain" in completed.stderr
        with pytest.raises(SpecError): run(Spec(**spec),d)


@pytest.mark.parametrize("json_output",[False,True],ids=["summary","json"])
def test_skill_cli_reports_do_not_echo_raw_outlier_values(tmp_path,json_output):
    # A unique synthetic sentinel appears in no requested summary statistic.
    d=pd.DataFrame({"y":[*np.linspace(-1,1,20),123456.789]})
    spec=tmp_path/"spec.json"; spec.write_text(json.dumps({"test":"one_sample_t","variables":{"outcome":"y"}}))
    data=tmp_path/"data.csv"; d.to_csv(data,index=False)
    result=cli("run",spec,data,*( ["--json"] if json_output else []))
    assert result.returncode==0,result.stderr
    assert "123456.789" not in result.stdout, "raw observation disclosed by normal skill run command"


def test_skill_check_rejects_incomplete_judgment_decisions(tmp_path):
    path=tmp_path/"incomplete.json"
    path.write_text(json.dumps({"test":"lmm","variables":{"outcome":"y","predictors":["visit"],"subject":"id"},"decisions":{"re_formula":"1"}}))
    result=cli("check",path)
    assert result.returncode!=0, "skill's validation step accepted a missing REML decision"


def test_skill_show_exposes_conditional_exploratory_decisions():
    result=cli("show","factor_sem")
    assert result.returncode==0
    for key in ["approach","n_factors","rotation","model_syntax"]:
        assert key in result.stdout, f"skill cannot discover required conditional key {key} via its show command"


def test_skill_positive_label_instruction_works_for_binary_leaf():
    d=pd.DataFrame({"response":["yes"]*7+["no"]*5})
    s=Spec.from_design({"goal":"single_outcome","outcome_type":"binary","structure":"independent","predictors":"none"},variables={"outcome":"response"},params={"positive_label":"yes","p0":.5})
    r=run(s,d)
    assert r.statistic==7
    assert r.p==pytest.approx(stats.binomtest(7,12,.5).pvalue)
