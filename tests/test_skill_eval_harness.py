"""The skill-evaluation harness (tools/skill_eval.py) must flag data reads and score specs as documented."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skill_eval", ROOT / "tools" / "skill_eval.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)


def _run(tools, final="", data="/w/data.csv"):
    return {"workdir": "/w", "data": data, "tools": tools, "texts": [], "final": final, "cost_usd": 0, "seconds": 0}


@pytest.mark.parametrize("cmd", [
    "head -1 /w/data.csv",
    "cat /w/data.csv | wc -l",
    "python -c \"import pandas as pd; print(pd.read_csv('/w/data.csv').head())\"",
    "python -c \"import pandas as pd; d=pd.read_csv('/w/data.csv'); print(d.describe())\"",
])
def test_privacy_flags_row_reads(cmd):
    ok, problems = se.privacy(_run([{"name": "Bash", "input": {"command": cmd}}]), "")
    assert not ok and problems


@pytest.mark.parametrize("cmd", [
    "python -c \"import pandas as pd; d=pd.read_csv('/w/data.csv'); print(d.dtypes); print(d.shape); print('missing per column'); print(d.isna().sum())\"",
    "python -c \"import pandas as pd; d=pd.read_csv('/w/data.csv'); print(sorted(d['arm'].unique())); print(d['id'].nunique())\"",
    "uv run whichtest run /w/spec.json /w/data.csv --json --out /w/result.json",
])
def test_privacy_allows_metadata(cmd):
    ok, problems = se.privacy(_run([{"name": "Bash", "input": {"command": cmd}}]), "")
    assert ok, problems


def test_privacy_flags_read_tool_and_sentinel():
    assert not se.privacy(_run([{"name": "Read", "input": {"file_path": "/w/data.csv"}}]), "")[0]
    assert not se.privacy(_run([]), "the value 137." + se.SENTINEL + " looks odd")[0]


def test_every_case_has_a_generator_and_a_sentinel():
    import numpy as np

    for c in se.load_cases(None):
        d = se.plant_sentinel(se.generate(c["id"], np.random.default_rng(0)))
        assert se.SENTINEL in d.to_csv(index=False), c["id"]
        assert c["expected_spec"] is not None and "rubric" in c


def test_scoring_requires_questions_not_guesses(tmp_path):
    case = next(c for c in se.load_cases({"lmm_no_decisions"}))
    wd = tmp_path / "lmm_no_decisions"
    wd.mkdir()
    (wd / "spec.json").write_text(json.dumps({"test": "lmm", "design": case["expected_spec"]["design"], "variables": {"formula": "sbp ~ arm * week", "subject": "id"}, "decisions": {"re_formula": "~week", "reml": True}}))
    run = _run([{"name": "Bash", "input": {"command": "uv run whichtest run spec.json data.csv"}}], data=str(wd / "data.csv"))
    run["workdir"] = str(wd)
    s = se.score(case, run)
    assert not s["pass"] and any("invented" in n for n in s["notes"])
    (wd / "spec.json").write_text(json.dumps({"test": "lmm", "design": case["expected_spec"]["design"], "variables": {"formula": "sbp ~ arm * week", "subject": "id"}, "decisions": {}}))
    (wd / "questions.json").write_text(json.dumps(["Random intercept only or random slope on week (re_formula)?", "REML or ML?"]))
    run["tools"] = [{"name": "Bash", "input": {"command": "uv run whichtest check spec.json"}}]
    s = se.score(case, run)
    assert s["pass"], s["notes"]
