# Statistical test pipeline

Give it a study design and your data; it picks the right test, checks the assumptions, runs it in Python, and cross-checks the answer with a second library.

## How it's built

- **Pipeline** (`whichtest` package): plain Python that runs a test from an explicit spec. Use it directly when you already know the test.
- **Skill** (`.claude/skills/whichtest/`): the reasoning layer on top. Works out which test fits, records why, and writes the spec. The instructions are model-agnostic, so a local model can run them when the study is private. `whichtest ask` is the deterministic stand-in.
- **Cross-check**: every test is re-run a second way. For most tests that is an independent library; for some it is a neighbouring statistic or a different modelling assumption (the result says which). Disagreement, or a cross-check that could not run, flags the result.

## Quick start

```bash
uv sync --extra dev            # or: pip install -e '.[dev]'  (dev includes the cross-check libraries)
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

## Install the skill in Claude Code

The skill is a folder, `.claude/skills/whichtest/`, holding `SKILL.md` and a generated question reference. Claude Code loads skills from a project's `.claude/skills/` and from your personal `~/.claude/skills/`. The skill shells out to the `whichtest` command, so install the package first, then put the folder where Claude Code will find it.

**1. Install the package** so `whichtest` is on your PATH (pick one):

```bash
uv tool install "whichtest[cross] @ git+https://github.com/bitmman/whichtest.git"   # isolated CLI, recommended
pip install "whichtest[cross] @ git+https://github.com/bitmman/whichtest.git"        # into the current environment
```

Append `@v0.1.0` to the URL to pin the released version. `[cross]` adds the independent libraries used for cross-checks (pingouin, lifelines, semopy, ...). Without it every test still runs, but each result says its cross-check was not run and is flagged.

**2. Install the skill** (pick one):

```bash
git clone https://github.com/bitmman/whichtest.git
cp -r whichtest/.claude/skills/whichtest ~/.claude/skills/          # for every project you open in Claude Code
cp -r whichtest/.claude/skills/whichtest /path/to/your/project/.claude/skills/   # for one project only
```

If you work inside this repository, nothing to copy: Claude Code picks the skill up from `.claude/skills/` here, and `uv run whichtest` stands in for an installed `whichtest`.

**3. Use it.** Start Claude Code in the folder with your data and describe the study ("two-arm trial, blood pressure at week 12 adjusted for baseline, which test?"), or type `/whichtest`. The skill asks only the taxonomy's questions, writes `spec.json`, validates it with `whichtest check`, runs `whichtest run`, and reports. For judgment tests (mixed models, GEE, factor analysis, competing risks, ICC, factorial ANOVA) it stops and asks for your modelling decisions instead of choosing them. It inspects your data by column names and dtypes only; no data rows are read into the model's context, and nothing but column names and the design leaves the machine.

**Other models.** `SKILL.md` is plain Markdown with no Claude-specific syntax. Any agent that can run shell commands can follow it; a local model can run it when the study is private.

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
- The 12 judgment tests refuse to run until `spec.decisions` carries the human choices (`uv run whichtest show lmm` lists them with their accepted values), reject look-alikes (`"false"`, `2.9`, unknown keys) instead of coercing them, and echo the choices verbatim in the result. `uv run whichtest check spec.json` validates all of this before any data is read.
- Skill written; not yet exercised on a real study.
- Continuous integration runs the suite on Python 3.10 to 3.13 with both newest and oldest allowed dependencies, plus a base-only install (no cross-check libraries) and a wheel install. Without the cross-check libraries every test still runs and says its cross-check was not run.
- Three rounds of third-party review (2026-10-12) found 13 + 9 + 8 defects, all fixed and pinned by tests. Rounds two and three contributed 174 independent acceptance tests (`tests/test_independent_*.py`, documented in `tests/INDEPENDENT_TESTS.md`) with reference values computed outside the package, and three study descriptions with target specs for evaluating the skill (`tests/fixtures/r3_study_cases.json`). See the review log in CLAUDE.md.
- Reports never contain individual observations: assumption checks report counts and extreme statistics, not values, so the pipeline's output can be read by a model without leaking data.
- The skill itself has been run through a model: `uv run python tools/skill_eval.py` drives headless Claude Code on eleven synthetic study descriptions and scores the specs it writes (routing, bindings, decisions asked rather than invented, no data rows read). Latest run: 11/11 (`tests/fixtures/skill_eval_latest.json`).
- One-sided alternatives are accepted only by the tests that can honour them end to end; the others reject the request instead of quietly reporting a two-sided p-value.

Next: use it on a real study. Anything the skill gets wrong there becomes a new case in `tests/fixtures/skill_eval_cases.json`.
