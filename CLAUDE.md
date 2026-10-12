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
  - `cli.py` — `list`, `show`, `resolve`, `ask` (terminal wizard), `tree`, `check`, `run`.
- `.claude/skills/whichtest/SKILL.md` — the skill (Claude Code project skill; the text is model-agnostic). `reference/questions.md` is generated: `uv run whichtest tree --out .claude/skills/whichtest/reference/questions.md`.
- `tests/` — pytest. `test_procedures.py` runs every ready and helper test on one synthetic dataset and asserts the cross-check agrees; `test_judgment.py` checks every judgment test refuses without decisions and runs with them; the exact identities listed under "Verify before trusting" are separate tests.
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
- `params` carries test constants: `mu0`/`m0`, `p0`, `k`/`n`, `table`, `observed`/`expected`, `count1`…`exposure2`, `weights` and `levels` (kappa), `positive_label` / `test_positive_label`, `timepoints`, `exact`, `zero_method`, `n_boot`, `seed`. Counts, exposures and tables are validated before use (`data.validate_counts`).
- `alternative` is honoured only by procedures registered with `directional=True`; `run()` rejects a one-sided request for the others rather than report a two-sided p-value under a one-sided label.
- `decisions` is where judgment tests get their human choices (random-effects structure, working correlation, ICC form, sums-of-squares type, estimand). `run()` refuses a judgment test whose declared decision keys are empty.
- The fields after the blank line are informational. `Spec.complete()` fills them from the taxonomy; if supplied they must agree or the spec is rejected.

## Result contract

`run()` returns a `TestResult` (see `result.py`): `statistic`, `p`, `df`, `effect_size`, `estimates[]` (with CIs), `checks[]` (`passed` is `None` for design properties only the human can confirm), `cross_check` (method, kind, agrees, per-quantity comparisons), `fallback` (filled from the taxonomy when a check fails), `post_hoc`, `warnings`, `extra`. `result.flagged` is true when the cross-check could not run or disagrees, any check failed, or the statistic / p-value is not finite (`inference_failed`). `significant` is `None` when there is no usable p-value. `summary()` for humans, `to_json()` for machines.

Conventions inside procedures:
- Run the taxonomy's `run` call as the primary and its `cross` call as the secondary; never reimplement what a library provides.
- Be honest about what a cross-check is. `kind="exact"` against a second library is an independent re-implementation. `kind="approximate"` or `"direction"` against another statistic, assumption or working correlation in the same library is a sensitivity comparison, and the `why` text must say so. The `result.cross_check.method` string names what was compared.
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

**Verified 2026-10-12, judgment tests:** all 12 run. Deviations found:

- `smf.ordinal_gee('y ~ x', ...)` adds an intercept that collides with the thresholds (parameters blow up to 1e14). Use the array interface `sm.OrdinalGEE(y, X_without_intercept, groups, cov_struct=...)`.
- `sm.NominalGEE` contrasts each category with the **last**; `mnlogit` with the **first**. Re-base before comparing directions.
- `ZeroInflatedNegativeBinomialP` fails to converge (NaN SEs) when the data aren't overdispersed (α on the boundary). Fit Nelder–Mead from ZIP start values, then refine with BFGS and keep the NM fit if BFGS returns non-finite SEs.
- `phreg(...).fit(groups=...)` and `phreg(..., strata=...)` match `lifelines.CoxPHFitter(cluster_col=...)` / `strata=[...]` to 1e-6.
- `pingouin.intraclass_corr` labels rows `ICC(1,1)`, `ICC(A,1)`, `ICC(C,1)`, `ICC(1,k)`, `ICC(A,k)`, `ICC(C,k)` in a fixed order; the CI column is `CI95`. The Shrout–Fleiss formulas from the two-way ANOVA mean squares reproduce all six.
- `statsmodels.multivariate.factor.Factor` returns complex-typed loadings (`np.real` them); `rotate()` works in place and returns None. pingouin has no KMO; it is computed in `multivariate.py`.
- `mixedlm.fit(method=[...])` accepts a list of optimizers to try in turn; random slopes on data without slope variance still end on the boundary, which the `convergence` check reports.

Still unverified:

- semopy (confirmatory factor analysis): not installed, not exercised; the `confirmatory` branch of `factor_sem` is untested.
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
5. ✅ The skill: `.claude/skills/whichtest/SKILL.md` walks the taxonomy questions, records its reasoning, and emits a spec. `reference/questions.md` beside it is generated by `whichtest tree` (a test fails if it drifts).
6. ✅ The 12 `judgment` tests, gated on explicit human-decision fields in the spec (`procedures/mixed.py` for lmm and the GEEs; the rest in their family modules). `run()` copies `spec.decisions` into `result.extra["decisions"]` so every judgment result shows what it was run with.

### Judgment decision keys

| test | `spec.decisions` | extra variable roles |
| --- | --- | --- |
| `factorial_anova` | `ss_type` (2 or 3), `include_interaction` (bool) | `predictors` = the factors |
| `lmm` | `re_formula` (`"1"` intercept only, `"~time"` adds a slope), `reml` (bool) | `subject` |
| `gee_cont`, `gee_logit`, `gee_poisson` | `working_correlation` (independence, exchangeable, ar1, unstructured) | `subject`; `exposure` for Poisson |
| `ordinal_gee`, `nominal_gee` | `working_correlation` (global_odds_ratio, independence) | `subject` |
| `zip` | `count_distribution` (poisson, negbin), `inflation_formula` (RHS, `"1"` = intercept only) | |
| `competing` | `estimand` (cumulative_incidence, cause_specific_hazard), `event_of_interest` (event code; 0 = censored) | `event` holds the event type |
| `cox_cluster` | `approach` (cluster_robust, stratify) | `cluster` or `strata` |
| `factor_sem` | `approach` (exploratory: + `n_factors`, `rotation`; confirmatory: + `model_syntax`, needs semopy) | `columns` = items |
| `icc` | `icc_form` (ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k, or the ICC(2,1)/ICC(A,1) spellings) | `columns` or `outcome`+`subject`+`within` |

### Adding a test

1. `uv run whichtest show <test_id>` for the taxonomy's run / cross / checks / report.
2. Write `fn(df, spec) -> TestResult` in the right `procedures/` module, decorated `@procedure("<test_id>", requires=(...))`. For a judgment test, also pass `decisions=(...)` naming the keys the human must set.
3. Fill `checks` (one `CheckResult` per taxonomy check key, in order) and `cross_check` via `compare()`.
4. Add a spec to `SPECS` in `tests/test_procedures.py`; the parametrized test asserts the cross-check agrees.

## Review log

**2026-10-12, third-party review (`../codex_astra_review.md`, 13 findings, all reproduced and fixed the same day; regression tests in `tests/test_review_findings.py`).** What changed, in the order the review listed them:

1. Binary labels are compared as typed values (`data.binary_labels`): `1.0`, `True` and `1` are the same positive; missing labels are excluded with one joint mask across columns and the exclusion is reported. The test column needs its own `test_positive_label` when it isn't 0/1.
2. `ordinal_logit` reports statsmodels' own LR test (`llr`, `llr_pvalue`) against the thresholds-only null; the hand-rolled version re-used the full model's thresholds and inflated the statistic 10×.
3. `poisson_2indep` applies one validity mask across count, exposure and group, validates counts (non-negative integers) and exposures (positive, finite), and reports exclusions.
4. Kaplan–Meier timepoints and the crossing check evaluate the right-continuous step function (`_surv_at`), not a linear interpolation.
5. `compare()` treats any non-finite value as disagreement; `direction` compares signs strictly (0 ≠ −1). `TestResult.significant` is `None` and `flagged` is true when p or the statistic is NaN (`inference_failed`).
6. ICC reports the F test and degrees of freedom of the chosen form (one-way ANOVA for ICC1/ICC1k, two-way otherwise).
7. ICC confidence intervals are computed for the requested level (McGraw & Wong 1996 formulas, matching pingouin's 95% row to 2 decimals); Aalen–Johansen gets `alpha=1-ci_level`.
8. GEE accepts `working_correlation="unstructured"` when `variables.time` gives integer visit positions; rows are sorted by subject and time so AR(1) has a defined order.
9. Competing-risk event codes are normalised once (numeric codes compared as numbers; `0` is censoring and is rejected as the event of interest).
10. `mnlogit` drops incomplete rows before category coding.
11. `summary()` prints `inf`/`-inf`/`NA` instead of crashing on one-sided intervals.
12. A cross-check that cannot run (missing optional library) becomes an explicit failed `CrossCheck` with a warning, so `flagged` and `--strict` see it. scikit-learn moved to the required dependencies because `diag_binary`, `roc_auc` and `delong` use it for the primary computation.
13. `SpecError` is imported at module scope in `procedures/parametric.py`.

Also from the review: the HTML drift test now parses the `NODES`/`LEAVES` literals in the finder page (via `json5`) and compares every node, option, routing target and test field with `taxonomy.json`; `zip`'s cross-check compares count-part coefficient directions with plain negative binomial instead of forcing agreement.

**Not done yet from the review:** typed, procedure-specific validation of `params` and `decisions` (they are still free dicts); a CI matrix across dependency versions and a base-only install test; an executed example for the confirmatory `factor_sem` branch.

**2026-10-12, round-two review (`../codex_astra_review_round2.md`, 9 findings) and independent acceptance suite** (`tests/test_independent_{exact,accuracy,models,contracts}.py`, 107 cases written by the reviewer against the public `run()` API with reference values from exact enumeration, sufficient statistics, pairwise DeLong placements and risk sets; `tests/INDEPENDENT_TESTS.md` documents them).

Runtime results recorded in this session: existing suite at `e7135ed` 142 passed; independent suite before fixes **76 passed, 31 failed**, every failure mapping to R1–R9; after the fixes below **249 passed, 0 failed** for the whole suite (Python 3.12.12, NumPy 2.5, pandas 3.0, SciPy 1.18.1, statsmodels 0.15.0, pingouin 0.7.0, lifelines 0.30.0). No reviewer test was skipped, marked xfail or weakened; one reviewer test was not changed but its contract was chosen among the alternatives it allowed (R6, below).

- R1 `mnlogit`: `pd.Categorical(...).remove_unused_categories()` after complete-case selection, so a declared-but-absent level can't become the reference.
- R2 `result.py`: `inference_failed` and `significant` recognise NaN/inf in any real scalar type (`_non_finite`, via `float()`); `compare()` with nothing to compare returns `agrees=False` ("nothing was compared").
- R3 `data.binary_labels`: the positive level need not be observed. A 0/1 column accepts 0 or 1 as positive whether or not it occurs; a labelled column with two observed levels must contain the declared positive. An all-negative prediction is a valid diagnostic test (sensitivity 0, specificity 1).
- R4 `data.formula_columns`: complete-case selection uses the identifiers a formula names, not substring matches; used by `mnlogit` and `lmm`.
- R5 `data.validate_counts` / `int_param` / `positive_float_param`: summary counts (`k`, `n`, `count1`…), exposures and `params.table` are validated for finiteness, integrality and range *before* any cast, for every table-taking procedure; `event_of_interest` must be an integer code.
- R6 one-sided alternatives are now a declared capability: `@procedure(..., directional=True)` marks the 13 procedures that apply `spec.alternative` consistently in primary and cross-check (t-tests, Wilcoxons, Mann–Whitney, Pearson, Spearman, partial correlation, simple regression, binomial, rate comparison, 2×2). `run()` rejects a one-sided request for any other test instead of silently reporting two-sided results. `simple_lin_reg` derives a one-sided secondary p from the OLS t; `partial_corr` computes the requested tail and passes `alternative` to pingouin; on a 2×2 table a one-sided request makes Fisher's exact test the primary (the chi-square has no direction), announced in `warnings`, with the chi-square kept in `extra`.
- R7 `Table(t, shift_zeros=False)` / `Table2x2(t, shift_zeros=False)` for the Pearson statistic and standardized residuals, so the cross-check tests the observed table including zero cells; a zero cell gets the 0.5 shift only for the odds-ratio / risk-ratio intervals, and the estimate's note says so.
- R8 `blocks()` and `paired()` take `numeric=False` for nominal procedures: Cohen's and Fleiss' kappa, Stuart–Maxwell and McNemar keep string labels through complete-case selection and code them afterwards (`categorical._sorted_levels`). Weighted kappa on string labels needs `params.levels` (an ordered list).
- R9 AUC with fewer than two cases in a class: `delong.py` raises; `roc_auc` returns the point estimate with no CI, no z/p, a failed `ci_method` check and a warning (so it is flagged); `delong` rejects with a `SpecError`.

Still open after round two: typed validation of `params`/`decisions` as a whole (the count/exposure/table/event-code inputs are now validated, but the dicts are still free-form); a CI matrix and base-only install; the confirmatory `factor_sem` branch.

## Conventions

- Tooling: `uv` (`uv sync --extra dev`, `uv run pytest`, `uv run whichtest ...`). The `cross` extra holds the independent libraries used only in cross-checks and post-hoc helpers; procedures import them lazily and degrade (no cross-check, noted) if absent.
- Keep two docs per project: `README.md` (short, for the human) and `CLAUDE.md` (detailed, for agents). Update both.
- Never send raw study data to any external service. Synthetic data only in tests and examples.
