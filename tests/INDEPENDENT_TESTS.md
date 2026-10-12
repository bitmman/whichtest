# Independent review suite — Claude Code handoff

## Round three — current handoff

Review: [`../../codex_astra_review_round3.md`](../../codex_astra_review_round3.md).
Initial baseline: `d943c39`; final integration: `93c02be`, which another session committed during the review.

**Executed by the reviewer:**

- Existing suite first, before authoring tests: **249 passed, 1 warning** (36.04s).
- New round-three suite: **52 passed, 15 failed** (49.52s), **67 cases**, zero skips/xfails.
- Final full suite on `93c02be`: **358 passed, 15 failed, 1 warning** (83.40s), zero skips. The concurrent commit added 57 cases: 53 base-install contracts and 4 CFA tests. All 306 original/concurrently added cases passed; the 15 failures are from this review.
- Final focused verification after strengthening JSON/text CI and individual assumption-marker checks: **4 passed, 7 deselected** (26.72s; `tests/test_independent_r3_cli.py -k every_example`).
- All four example specs ran through actual CLI subprocesses with JSON/text × strict/non-strict; all four passed numerical and output-consistency checks. The factorial example returns strict exit 2 for unbalanced cells; the other three return 0.

Only test modules, study fixtures, this inventory, and the parent review were authored here. Production source and prior tests were not edited by the reviewer. Failures are ordinary acceptance failures retained for remediation; do not add skip/xfail markers.

**Status after remediation (Claude Code session, 2026-10-12):** all 15 failures reproduced and mapped to R3-1…R3-8; fixes are listed in CLAUDE.md's review log. Round-three suite: **67 passed**; whole suite: **396 passed** (the 67 here, 23 own regression tests in `tests/test_round3.py`, and the 306 earlier cases) on Python 3.12 with current dependencies, and again on Python 3.10 with the lowest allowed dependency versions. No test in this suite was changed, skipped or marked xfail.

### Run round three

From `stats-test-pipeline/`, using the development/cross environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/whichtest-r3-mpl \
  .venv/bin/python -m pytest tests/test_independent_r3_*.py -ra --tb=short

# Then integrate with every existing test.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/whichtest-r3-mpl \
  .venv/bin/python -m pytest -ra --tb=short
```

The environment used NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, statsmodels 0.15.0, lifelines 0.30.0 and pytest 9.1.1. semopy 2.3.11 was installed by the concurrent session before the final full run. The new tests require lifelines directly for external Cox references; missing it is not silently skipped. No new dependency declaration was added.

### Round-three inventory

| Module / fixture | Cases | Independent evidence |
| --- | ---: | --- |
| `test_independent_r3_judgment.py` | 53 | 16 numerical/decision-effect cases, 18 omitted-key gates, 19 invalid-decision cases. Balanced REML and ICC mean-square identities; grouped GEE means/odds/rates; direct statsmodels model fits; lifelines robust/stratified/cause-specific Cox; hand Aalen–Johansen risk sets. All 12 judgment procedures covered, including factorial ANOVA and both requested survival branches. |
| `test_independent_r3_cli.py` | 11 | Four example specs with direct primary-statistic references and JSON/text/strict checks; accepted fallback/rejected mismatched design; two raw-data disclosure renderers; skill preflight/discovery and positive-label contracts. |
| `test_independent_r3_skill_fixtures.py` | 3 | Golden specs resolve and run through public APIs against paired-change arithmetic, direct OLS, and direct MixedLM references. |
| `fixtures/r3_study_cases.json` | 3 descriptions | Increasing difficulty: one-cohort paired change, adjusted parallel trial, longitudinal treatment×visit model. Exact target specs and evaluation rubrics for a future model harness. These are not measured LLM results. |

Every valid judgment reference run compares the exact supplied decisions dictionary with `result.extra["decisions"]` and checks it was not mutated. Numerical expectations never use whichtest's cross-check output. Only `Spec`, `SpecError`, `run`, and CLI subprocesses access whichtest; no private production helpers are imported. All data are synthetic, including the documented example trial.

Factor references have explicit initial values and compare loading-implied covariance to avoid arbitrary signs. Optimizer tolerances are specified in tests. The GEE correlation-effect fixture uses within-subject variation because balanced group-only models can legitimately yield identical estimates for multiple working correlations. AR(1)/unstructured correctness was not re-audited.

### Round-three failure map

| Review item | Failing cases | Test name / parameter fragment |
| --- | ---: | --- |
| R3-1 robust Cox headline p | 1 | `test_cox_approach_matches_lifelines_coefficients_and_intervals[cluster_robust]` |
| R3-2 raw outcome disclosure | 2 | `test_skill_cli_reports_do_not_echo_raw_outlier_values` |
| R3-3 factor choice gate/discovery | 2 | `test_judgment_missing_each_required_key_is_rejected[factor_sem-rotation]`; `test_skill_show_exposes_conditional_exploratory_decisions` |
| R3-4 silent decision coercion | 4 | `test_judgment_invalid_decision_is_spec_error`: `ss_type-2.9`, `include_interaction-false`, `reml-false`, `n_factors-1.5` |
| R3-5 LMM optimizer exception | 1 | `test_lmm_optimizer_recovers_from_singular_first_attempt` |
| R3-6 dependency exception leakage | 3 | `test_judgment_invalid_decision_is_spec_error`: missing random/inflation columns; invalid rotation |
| R3-7 misleading preflight OK | 1 | `test_skill_check_rejects_incomplete_judgment_decisions` |
| R3-8 skill positive-label mismatch | 1 | `test_skill_positive_label_instruction_works_for_binary_leaf` |

The skill's routing risks are explicitly labeled inspection findings in the report. The fixtures validate intended destinations and calculations; no LLM was run. Base-only behavior was assessed with blocked imports and the newly added project tests, not a clean installation. The final full run exercised the concurrent CFA tests; no new independent confirmatory acceptance tests were authored. The earlier missing-lifelines CIF behavior and confirmatory concerns were corrected by `93c02be` and are recorded as resolved.

## Round two — historical record

Prepared against commit `e7135ed` on 2026-10-12. Review: `../../codex_astra_review_round2.md`.

**Status: syntax and pytest collection checked by the reviewer; executed in the Claude Code session of 2026-10-12.** Baseline before fixes: 76 passed, 31 failed (all 31 mapped to R1–R9). After the fixes recorded in CLAUDE.md's review log: 107 passed, and the full suite is 249 passed. No test here was skipped, marked xfail or altered. One contract choice to note: for a one-sided alternative on a dense 2×2 table the pipeline runs Fisher's exact test as the primary (with a warning) rather than rejecting, which both `test_dense_two_by_two_does_not_silently_ignore_direction` and `test_fisher_matches_enumerated_fixed_margins` accept.

## Run

From the `stats-test-pipeline` project directory, using the existing development environment:

```bash
# Establish a baseline including the first review's regression tests.
.venv/bin/python -m pytest \
  --ignore=tests/test_independent_exact.py \
  --ignore=tests/test_independent_accuracy.py \
  --ignore=tests/test_independent_models.py \
  --ignore=tests/test_independent_contracts.py

# Run all new independent tests; retain every failure for triage.
.venv/bin/python -m pytest -ra \
  tests/test_independent_exact.py \
  tests/test_independent_accuracy.py \
  tests/test_independent_models.py \
  tests/test_independent_contracts.py

# After fixes, run the complete suite.
.venv/bin/python -m pytest
```

`uv run pytest` with the same arguments is also suitable after the normal development dependencies have been synchronized. No additional dependency is introduced by this suite. The cumulative-incidence oracle skips if lifelines is absent; use the project's dev/cross environment to exercise it. Other tests exercise missing optional cross-check behavior where appropriate.

## Inventory

| Module | Cases | Independent evidence |
| --- | ---: | --- |
| `test_independent_exact.py` | 28 | Binomial masses with exact fractions; Fisher fixed-margin hypergeometric enumeration; McNemar discordant outcomes; all Mann–Whitney label allocations; all signed-rank sign assignments; Pearson/Gof chi-square arithmetic; Friedman rank sums. |
| `test_independent_accuracy.py` | 16 | Asymmetric confusion matrix tally; nullable labels; joint exclusion; AUC pair counts; DeLong sample variances/covariance from pairwise placements; monotone-transform and score-swap identities; unestimable variance. |
| `test_independent_models.py` | 27 | One-sample and Welch sufficient statistics; paired alignment; ANOVA sums of squares; regression residuals orthogonal to the design; partial correlation identity; grouped logistic/Poisson likelihoods; ICC cell arithmetic; survival risk sets; multinomial equivalent-input checks. |
| `test_independent_contracts.py` | 36 | Finite/integer input contracts; event-code validation; nominal label renaming; directional options; non-finite result types; empty comparisons; strict CLI status and JSON persistence. |

## Independence and tolerances

- The numerical reference calculations do not import `whichtest` private helpers, its effect-size functions, or existing clinical/RNG fixtures. Public `run()` and the public DeLong functions are the subjects under test.
- Exact probability oracles use Python combinatorics and `Fraction`. Normal/chi-square special-case tails use elementary `erfc`/`exp` expressions.
- Student-t CDFs/quantiles come from SciPy, but sufficient statistics, degrees of freedom, and regression coefficients are independently calculated. These are independent fit checks, not an independent implementation of every distribution function.
- DeLong references explicitly compare every positive/negative score pair with half credit for ties, then compute placement variances. They do not use the production midrank algorithm or sklearn AUC.
- Model transformations compare specified outputs, not entire results: changing category names can legitimately change labels, and flags can reflect separate diagnostics. Cross-check agreement is asserted only where matching the same quantity is part of the tested contract.
- Small exact-test fixtures avoid tie/zero ambiguity and approximate-method switching. Each stochastic fixture has its own fixed seed; there is no statistical pass-rate/flakiness threshold.
- Tight tolerances apply to algebraic identities; optimizer outputs use pytest's ordinary approximate comparison. If a platform needs a different tolerance, justify it from optimization accuracy and retain the substantive assertion.

## Triage rules

1. Keep the first-review regression suite. The new suite adds independent evidence; it does not replace prior coverage.
2. Review each failure against the asserted mathematical or input contract. A broken oracle should be corrected with a derivation; production output alone is not a reference answer.
3. Tests expecting `SpecError` define a clear public validation boundary. Overflow, incidental `ValueError`, and silently truncated values are not equivalent outcomes.
4. Directional partial correlation and dense 2×2 tests allow an explicit `SpecError` for an unsupported alternative. AUC with an undersized class permits either explicit rejection or a flagged result without a misleading interval. These alternatives are deliberate.
5. Do not add blanket skips, xfails, relax checks, or hardcode fixtures to make the suite green. Record any intentional API change and update the review rationale accordingly.
6. Record baseline/new-suite/full-suite results, dependency versions, and the fixes made. Do not report collection success as test success.

## Review-to-test map

| Review item | Test name fragments |
| --- | --- |
| R1 categorical outcome labels | `multinomial_removes_unused_categorical_levels` |
| R2 verification gaps | `numpy_scalar`, `nonfinite_pvalue`, `empty_comparison` |
| R3 constant negative predictions | `constant_binary_prediction` |
| R4 formula dependency selection | `formula_ignores_unused_columns` |
| R5 validation | `fractional_summary_counts`, `poisson_summary_requires`, `contingency_tables_reject`, `event_of_interest_is_not_truncated` |
| R6 alternatives | `linear_regression_orthogonal`, `partial_correlation_honors`, `dense_two_by_two` |
| R7 zero-cell table modification | `pearson_table_statistic` |
| R8 nominal labels | `nominal_category_renaming` |
| R9 undefined AUC uncertainty | `auc_cannot_present_unestimable_variance` |
