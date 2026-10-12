# Independent review suite — Claude Code handoff

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
