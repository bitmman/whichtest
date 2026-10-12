# Statistical test pipeline

Give it a study design and your data; it picks the right test, checks the assumptions, runs it in Python, and cross-checks the answer with a second library.

## How it's built

- **Pipeline** (`whichtest` package): plain Python that runs a test from an explicit spec. Use it directly when you already know the test.
- **Skill** (`.claude/skills/whichtest/`): the reasoning layer on top. Works out which test fits, records why, and writes the spec. The instructions are model-agnostic, so a local model can run them when the study is private. `whichtest ask` is the deterministic stand-in.
- **Cross-check**: every test is re-run a second way. For most tests that is an independent library; for some it is a neighbouring statistic or a different modelling assumption (the result says which). Disagreement, or a cross-check that could not run, flags the result.

## Quick start

```bash
uv sync --extra dev            # or: pip install -e '.[dev]'
uv run pytest

uv run whichtest ask --out spec.json             # answer the taxonomy questions → spec
# edit spec.json: bind your columns under "variables"
uv run whichtest run spec.json data.csv          # report; add --json for the full result
uv run whichtest run examples/spec_ancova.json examples/trial.csv
```

From Python:

```python
import whichtest as w
spec = w.Spec.from_design(
    {"goal": "single_outcome", "outcome_type": "continuous", "structure": "independent", "predictors": "one_factor_2_levels"},
    variables={"outcome": "sbp_week12", "group": "arm"},
)
result = w.run(spec, df)       # TestResult: statistic, p, effect size, checks, cross-check, fallback
print(result.summary())
result.flagged                 # True if a check failed or the cross-check disagreed
```

## What's here

- `.claude/skills/whichtest/`: the skill (`SKILL.md`) and its generated question-tree reference.
- `src/whichtest/`: the package. `taxonomy.py` reads the tree, `spec.py` is the contract, `registry.py` dispatches, `procedures/` hold the tests.
- `stat-test-finder.html`: open in a browser. Click through the decision tree, or browse the full map.
- `taxonomy.json`: the same tree as data. This is what the code reads.
- `examples/`: a synthetic trial dataset and three specs.
- `CLAUDE.md`: full context for Claude Code: decisions, prior art, spec format, what to verify.

## Where it stands

- Taxonomy done: 51 tests across 54 decision paths.
- **All 51 tests implemented**, each with assumption checks, effect sizes, an independent cross-check, and a unit test (`uv run whichtest list`). Includes a DeLong implementation validated against a bootstrap.
- The 12 judgment tests refuse to run until `spec.decisions` carries the human choices (`uv run whichtest show lmm` lists them), and echo those choices in the result.
- Skill written; not yet exercised on a real study.
- Two rounds of third-party review (2026-10-12) found 13 + 9 defects, all fixed and pinned by tests. The second round also contributed 107 independent acceptance tests (`tests/test_independent_*.py`, documented in `tests/INDEPENDENT_TESTS.md`) with reference values computed outside the package; the whole suite is 249 tests. See the review log in CLAUDE.md.
- One-sided alternatives are accepted only by the tests that can honour them end to end; the others reject the request instead of quietly reporting a two-sided p-value.

Next: try it on a real (synthetic-stand-in) analysis end to end, then tighten what the skill asks.
