import json
from pathlib import Path

import pytest

from whichtest.taxonomy import TaxonomyError, load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def test_counts():
    t = load_taxonomy()
    assert len(t.tests) == 51
    assert len(t.questions) == 17
    assert len(list(t.paths())) == 54
    assert len(t.by_status("ready")) == 28
    assert len(t.by_status("helper")) == 11
    assert len(t.by_status("judgment")) == 12


def test_every_test_reachable_and_every_leaf_known():
    t = load_taxonomy()
    reached = {tid for tid, _ in t.paths()}
    assert reached == set(t.tests)


def test_resolve_and_next_question():
    t = load_taxonomy()
    d = {"goal": "single_outcome", "outcome_type": "continuous", "structure": "independent", "predictors": "one_factor_2_levels"}
    assert t.resolve(d) == "indep_t"
    assert t.next_question({"goal": "single_outcome"}).key == "outcome_type"
    assert t.next_question(d) is None
    assert t.path_keys(d) == ["goal", "outcome_type", "structure", "predictors"]


def test_resolve_errors():
    t = load_taxonomy()
    with pytest.raises(TaxonomyError):
        t.resolve({"goal": "single_outcome"})
    with pytest.raises(TaxonomyError):
        t.resolve({"goal": "nonsense"})


def test_multi_path_tests():
    t = load_taxonomy()
    assert len(t.paths_to("spearman")) == 2
    assert len(t.paths_to("chi_sq_indep")) == 3


def test_html_and_json_do_not_drift():
    """The finder page embeds the same taxonomy; the test ids and statuses must match."""
    html = (ROOT / "stat-test-finder.html").read_text(encoding="utf-8")
    t = load_taxonomy()
    import re

    for tid, info in t.tests.items():
        assert re.search(rf"\b{tid}\b", html), f"{tid} missing from stat-test-finder.html"
        assert info.name.replace("'", "\\'") in html or info.name in html, f"{tid} name differs in stat-test-finder.html"
    raw = json.loads((ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    assert set(raw["tests"]) == set(t.tests)
