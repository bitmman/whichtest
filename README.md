# Statistical test pipeline

Give it a study design and your data; it picks the right test, checks the assumptions, runs it in Python, and cross-checks the answer with a second library.

## How it's built

- **Pipeline** (`whichtest` package): plain Python that runs a test from an explicit spec. Use it directly when you already know the test.
- **Skill**: the reasoning layer on top. Works out which test fits and writes the spec. Can run on a local model when the study is private. (Not written yet; the `whichtest ask` wizard is the deterministic stand-in.)
- **Cross-check**: every test is re-run through an independent implementation. If they disagree, the result is flagged.

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

- `src/whichtest/`: the package. `taxonomy.py` reads the tree, `spec.py` is the contract, `registry.py` dispatches, `procedures/` hold the tests.
- `stat-test-finder.html`: open in a browser. Click through the decision tree, or browse the full map.
- `taxonomy.json`: the same tree as data. This is what the code reads.
- `examples/`: a synthetic trial dataset and three specs.
- `CLAUDE.md`: full context for Claude Code: decisions, prior art, spec format, what to verify.

## Where it stands

- Taxonomy done: 51 tests across 54 decision paths.
- **39 of 51 tests implemented**: all 28 ready and all 11 helper tests (including a DeLong implementation validated against a bootstrap), each with assumption checks, effect sizes, a cross-check, and a unit test (`uv run whichtest list`).
- 12 judgment tests are still to do. They will refuse to run until `spec.decisions` is filled in.
- No skill yet.

Next: the skill, then the 12 judgment tests.
