# Statistical test pipeline — agent context

Detailed context for coding agents. The short human version is `README.md`; keep both in sync when either changes.

## What this project is

A Python library that takes a study design (and, locally, the dataset) and:

1. picks the appropriate statistical test,
2. checks that test's assumptions,
3. runs it with existing libraries (scipy, statsmodels) instead of reimplementing,
4. cross-checks the result with an independent second implementation.

Motivation: test selection for complicated, multi-dimensional clinical datasets involves a lot of back-and-forth. The goal is to automate the clear cases and flag the ambiguous ones for a human.

Design work so far happened in a claude.ai chat on 2026-10-12. The interactive taxonomy from that chat is `stat-test-finder.html` (also published privately at https://claude.ai/artifact/VZNb8N1Kr2i8rdAPBSkpQw).

## Decisions already made

- **Skill wraps pipeline.** Two layers:
  - **Pipeline** — deterministic, reproducible Python. Takes an explicit spec and runs it. Usable on its own when the test is already known.
  - **Skill** — the reasoning layer. Turns a description of the design into a spec, and handles ambiguity.
- **Clean contract between layers.** The pipeline knows nothing about reasoning; the skill's only output is the spec. Either side can change without touching the other.
- **Swappable reasoning model.** Frontier model when describing the design to it is acceptable; local model when the design or data must stay private.
- **Privacy.** Study data and designs are often private before publication. Raw data never leaves the machine. At most, abstract design metadata (outcome type, structure, number of groups) goes to a cloud model, and only when the user chooses that.
- **Python only for now.** Core calls use scipy and statsmodels. Other packages (pingouin, lifelines, scikit-learn, scikit-posthocs, semopy, krippendorff) appear only where noted — mostly in cross-checks, where an independent library is the point. R is out of scope for now.
- **Cross-validation is built in, not optional.** Especially early on, nothing should be trusted blindly:
  - the skill outputs its reasoning (checks run, alternatives ruled out), not just the chosen test;
  - every test has a second implementation; if the two disagree beyond tolerance, flag it for a human.
- **Coverage is the whole taxonomy**, not just multi-dimensional cases. The taxonomy below is the coverage map.

## Prior art (literature review, 2026-10-12)

- **Decision-tree selectors** — UCLA OARC "What statistical analysis should I use?" (the canonical table; NYU and others adapt it), Statkat (also a jamovi module), Laerd Statistics, Stat Tree, INSPECT-LB. They recommend a test; they don't run it on your data.
- **ChooseMyStat** (medRxiv, 2026) — web tool for clinical research that generates an analysis plan from study design; validated on 105 published articles across 15 specialties. Closest to our framing, but advisory and web-based.
- **AIPOCH medical research skills** — an agent-skill suite covering academic writing, data analysis, protocol design and evidence synthesis. Statistics is one thread, not the focus, so it doesn't go deep on test selection.
- **Gap we target:** local-first, design + data → test → assumption checks → run → cross-check.

## Files

- `src/whichtest/` — the package (`uv sync --extra dev`; `uv run pytest`; `uv run whichtest --help`).
  - `taxonomy.py` — loads `taxonomy.json` (repo root in a checkout, packaged copy in a wheel), validates it, walks it: `next_question`, `resolve`, `paths_to`.
  - `spec.py` — pydantic `Spec` / `Design` / `Variables`. `Spec.from_design()` resolves through the taxonomy; `Spec.complete()` fills the informational fields and refuses inconsistent specs (a `test` that differs from what the design resolves to is allowed only if it is on that test's fallback chain).
  - `result.py` — `TestResult`, `CheckResult`, `Estimate`, `CrossCheck`; `compare()` implements the three agreement kinds (`exact` within tolerance, `approximate` = same decision at alpha or |Δp| small, `direction` = same sign).
  - `registry.py` — `@procedure(test_id, requires=..., decisions=...)` decorator and `run(spec, df)`; judgment tests refuse to run unless `spec.decisions` has every key the procedure declares.
  - `data.py` — column binding helpers (`groups`, `paired`, `blocks`, `contingency_table`). Wide and long repeated-measures layouts are both accepted.
  - `checks.py`, `effects.py` — shared assumption checks and effect sizes.
  - `delong.py` — DeLong AUC variance and paired AUC comparison (not in any library).
  - `procedures/` — one module per family (`multivariate.py` holds MANOVA / MANCOVA / multivariate regression). Each procedure is `fn(df, spec) -> TestResult` and must set `checks` and `cross_check`.
  - `cli.py` — `list`, `show`, `resolve`, `ask` (terminal wizard), `check`, `run`.
- `tests/` — pytest. `test_procedures.py` runs every ready and helper test on one synthetic dataset and asserts the cross-check agrees; the exact identities listed under "Verify before trusting" are separate tests.
- `examples/` — synthetic `trial.csv` and three specs.
- `taxonomy.json` — the full decision tree as data. **Source of truth for test selection.** Large (~80 KB): read it with tools when needed rather than importing it into context.
  - `nodes`: question id → `{ q, key, opts }`. Each option has `val` (the spec value for that answer) and `to` (next node id, or `L:<test_id>`).
  - `tests`: test id → `{ name, family, status, when, checks[], fb, run, call, cross, xcall, report, post?, watch? }`.
    - `checks[]`: `{ k, t, c? }` — assumption key, description, optional code.
    - `fb`: fallback if a check fails; `to` points to another test id, `k` names a technique not in the tree.
    - `cross`: `{ c, why }` — cross-check code and what agreement means.
  - `status_levels`: definitions of `ready`, `helper`, `judgment`.
- `stat-test-finder.html` — self-contained interactive version of the same tree (wizard + full map). The data in it is identical to `taxonomy.json`. If the taxonomy changes, regenerate one from the other; don't let them drift.

## Taxonomy summary

17 question nodes, 51 tests, 54 decision paths (some tests are reachable by more than one path).

Top-level question (`goal`):

| Answer | Next questions |
| --- | --- |
| `single_outcome` | `outcome_type` → `structure` → `predictors` / `repeated_pattern` / `aim` / `survival_aim` |
| `multiple_outcomes` | `multivariate_aim` |
| `association` | `variable_types` |
| `agreement_accuracy` | `task` |

Outcome types: continuous, ordinal, binary, nominal, count, time to event.
Structures: independent, paired, repeated/clustered (split further where it matters).

Readiness:

- **ready (28)** — one scipy or statsmodels call. Safe to automate first.
- **helper (11)** — needs a small function or another package: `rm_anova`, `ordinal_logit`, `cox`, `manova`, `mancova`, `mv_reg`, `partial_corr`, `bland_altman`, `diag_binary`, `roc_auc`, `delong`.
- **judgment (12)** — modeling decisions the pipeline must not make alone: `factorial_anova`, `lmm`, `gee_cont`, `ordinal_gee`, `gee_logit`, `nominal_gee`, `gee_poisson`, `zip`, `competing`, `cox_cluster`, `factor_sem`, `icc`.

## Spec contract

Implemented in `spec.py`. What the skill hands to the pipeline. `whichtest resolve key=value ...` prints a template for any path.

```json
{
  "test": "indep_t",
  "design": {"goal": "single_outcome", "outcome_type": "continuous", "structure": "independent", "predictors": "one_factor_2_levels"},
  "variables": {"outcome": "sbp_week12", "group": "arm"},
  "params": {},
  "alpha": 0.05,
  "alternative": "two-sided",
  "ci_level": 0.95,
  "decisions": {},
  "reasoning": ["Two independent arms, continuous outcome, no covariates."],

  "status": "ready",
  "call": "scipy.stats.ttest_ind(equal_var=False)",
  "assumption_checks": ["normality_by_group", "independence", "variance_info"],
  "fallback": "mann_whitney",
  "cross_check": "statsmodels.stats.weightstats.ttest_ind(usevar='unequal')"
}
```

- `variables` roles: `outcome`, `group`, `subject`, `within`, `columns` (wide repeated / raters), `before`/`after`, `x`/`y`, `row`/`col`, `time`, `event`, `exposure`, `predictors`, `categorical`, `formula`. A procedure declares which it needs (`requires=`); `"a|b"` means either.
- `params` carries test constants: `mu0`/`m0`, `p0`, `k`/`n`, `table`, `observed`/`expected`, `count1`…`exposure2`, `weights` (kappa), `timepoints`, `exact`, `zero_method`, `n_boot`, `seed`.
- `decisions` is where judgment tests get their human choices (random-effects structure, working correlation, ICC form, sums-of-squares type, estimand). `run()` refuses a judgment test whose declared decision keys are empty.
- The fields after the blank line are informational. `Spec.complete()` fills them from the taxonomy; if supplied they must agree or the spec is rejected.

## Result contract

`run()` returns a `TestResult` (see `result.py`): `statistic`, `p`, `df`, `effect_size`, `estimates[]` (with CIs), `checks[]` (`passed` is `None` for design properties only the human can confirm), `cross_check` (method, kind, agrees, per-quantity comparisons), `fallback` (filled from the taxonomy when a check fails), `post_hoc`, `warnings`, `extra`. `result.flagged` is true when the cross-check disagrees or any check failed. `summary()` for humans, `to_json()` for machines.

Conventions inside procedures:
- Run the taxonomy's `run` call as the primary and its `cross` call as the secondary; never reimplement what a library provides.
- Automatic switches are allowed only where the taxonomy's own `run` snippet lists them (2×2 → Fisher on sparse cells; Poisson → negative binomial on overdispersion). Say so in `warnings` and keep the original in `extra`.
- Anything else that fails a check becomes a `fallback` suggestion, not a silent change of test.

## Verify before trusting

The code snippets in `taxonomy.json` were written from library knowledge, not executed. Before relying on any of them, run them on synthetic data with known answers.

**Verified 2026-10-12** (Python 3.12, scipy 1.18.1, statsmodels 0.15.0, pingouin 0.7.0, lifelines 0.30.0, scikit-learn 1.9.1): every snippet for the 28 ready tests runs as written. Deviations found while implementing:

- `cochrans_q(..., return_object=True)` is deprecated; call it without the keyword.
- `scipy.stats.contingency.association` requires an integer table; round and cast first.
- scikit-learn ≥ 1.8: `LogisticRegression(penalty=None)` is deprecated; unpenalized is `C=np.inf`.
- `MNLogit.conf_int()` has an awkward index; the procedure uses `params ± z·bse` instead.
- GLM results have no `llr_pvalue`; compute the LR test from `llf` and `llnull`.
- `krippendorff` has no `__version__`.
- Pins: `scipy>=1.11` (CI on `ttest_ind` results), `statsmodels>=0.14`.

**Verified 2026-10-12, helper tests:** all 11 run. Deviations found:

- `OrderedModel.from_formula('y ~ 0 + ...')` raises "There should not be a constant" as soon as a categorical predictor is present (patsy full-codes it). Build the design with an intercept via `patsy.dmatrices` and drop the `Intercept` column instead.
- `pingouin.sphericity`, `rm_anova(correction=True, detailed=True)`, `box_m`, `partial_corr`, `multivariate_normality` all exist with the documented arguments. `partial_corr` returns the p-value as `p-val`.
- `phreg(..., ties='efron')` matches `lifelines.CoxPHFitter` to about 1e-5 (statsmodels stops its optimizer a little earlier); compare at `rtol=1e-4`.
- `statsmodels.graphics.mean_diff_plot` draws its limits with the **population** SD (`ddof=0`); the reported limits use `ddof=1`, so compare like with like.
- `sklearn.metrics.confusion_matrix` needs `labels=[0, 1]` or a single-class input collapses to a 1×1 matrix.

Still unverified (judgment tests):

- statsmodels `phreg(...).fit(groups=...)` for cluster-robust Cox standard errors.
- Signatures of `smf.ordinal_gee` and `smf.nominal_gee`.
- `pingouin.intraclass_corr`.
- Exact-agreement cross-checks (paired t vs one-sample t on differences, Spearman vs Pearson on ranks, AUC vs Mann–Whitney U / (n₁n₀), exact McNemar vs binomial on discordant pairs, logit vs binomial GLM, point-biserial vs Student's t) are unit tests in `tests/test_procedures.py` and pass.

Known holes in the Python ecosystem:

- DeLong paired-AUC test has no implementation in scipy, statsmodels or scikit-learn. Written in `delong.py` (fast DeLong, Sun & Xu 2014) and validated against a paired bootstrap.
- Fine–Gray subdistribution hazards: not in statsmodels or lifelines.
- Aligned rank transform (ART) ANOVA: no maintained Python port.

## Build order

Started 2026-10-12 on the user's go-ahead.

1. ✅ Spec schema in code (pydantic), validated against `taxonomy.json`.
2. ✅ Pipeline registry: `test_id → procedure`, each returning a structured result.
3. ✅ The 28 `ready` tests, each with a unit test where the cross-check must agree.
4. ✅ The 11 `helper` tests, including DeLong (`delong.py`, fast DeLong after Sun & Xu 2014; validated against sklearn's AUC and a stratified paired bootstrap in `tests/test_delong.py`).
5. ⬜ The skill: a SKILL.md that walks the taxonomy questions, records its reasoning, and emits a spec.
6. ⬜ The 12 `judgment` tests, gated on explicit human-decision fields in the spec.

### Adding a test

1. `uv run whichtest show <test_id>` for the taxonomy's run / cross / checks / report.
2. Write `fn(df, spec) -> TestResult` in the right `procedures/` module, decorated `@procedure("<test_id>", requires=(...))`. For a judgment test, also pass `decisions=(...)` naming the keys the human must set.
3. Fill `checks` (one `CheckResult` per taxonomy check key, in order) and `cross_check` via `compare()`.
4. Add a spec to `SPECS` in `tests/test_procedures.py`; the parametrized test asserts the cross-check agrees.

## Conventions

- Tooling: `uv` (`uv sync --extra dev`, `uv run pytest`, `uv run whichtest ...`). The `cross` extra holds the independent libraries used only in cross-checks and post-hoc helpers; procedures import them lazily and degrade (no cross-check, noted) if absent.
- Keep two docs per project: `README.md` (short, for the human) and `CLAUDE.md` (detailed, for agents). Update both.
- Never send raw study data to any external service. Synthetic data only in tests and examples.
