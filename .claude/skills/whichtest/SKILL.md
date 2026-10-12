---
name: whichtest
description: Pick the right statistical test for a study design, write the spec, run it with assumption checks and an independent cross-check, and explain the result. Use when someone describes a study, dataset or analysis question ("which test should I use", "compare groups", "is X associated with Y", "analyse this trial") or asks to run or interpret a test with the whichtest pipeline.
---

# whichtest: from study design to a checked result

You are the reasoning layer. The pipeline (`whichtest` CLI / Python package) is deterministic and knows nothing about the study; your only output to it is a **spec**. Keep that contract: do not compute statistics yourself, do not bypass the pipeline, do not edit the taxonomy to make a case fit.

## Privacy rules (non-negotiable)

- Never read, print or paste raw data rows. Work from column names, dtypes, and summary counts only (`head -1 data.csv`, `python -c "import pandas as pd; print(pd.read_csv('data.csv').dtypes)"`). Say so in your reasoning.
- The spec contains column names and design metadata. That is all that may leave the machine, and only if the user chose a cloud model.

## Workflow

1. **Elicit the design.** Ask only what the taxonomy needs, in the taxonomy's order (`reference/questions.md`, or `whichtest tree`). The questions are: goal → outcome type → structure → predictors / aim. Stop asking once a leaf is reached. If the user already told you an answer, don't ask again. Record each answer and the reason you chose it.
2. **Resolve the test.** `whichtest resolve goal=... outcome_type=... structure=... predictors=...` prints the spec template with the taxonomy's status, checks, fallback and cross-check. `whichtest show <test_id>` gives the full entry, the variable roles it needs, and (for judgment tests) the decision keys.
3. **Rule out neighbours.** Name the one or two tests you did *not* pick and why (e.g. "paired, not independent, because the same patients were measured twice"). This goes into `reasoning`.
4. **Bind the columns.** Look at the header and dtypes. Fill `variables` with the roles from `whichtest show`. Check binary columns are 0/1 (or set `params.positive_label`), events are 1 = event, repeated data are either wide (`columns`) or long (`outcome` + `subject` + `within`).
5. **Judgment tests.** If the status is `judgment`, the pipeline will refuse to run until `decisions` is filled. Do not fill it yourself. Present the options to the user with a plain recommendation and the trade-off, then record their choice. Examples: random-effects structure and REML for `lmm`; working correlation for GEE; sums-of-squares type for `factorial_anova`; ICC form; estimand for competing risks.
6. **Write the spec** to a file and validate: `whichtest check spec.json`.
7. **Run**: `whichtest run spec.json data.csv` (add `--json --out result.json` to keep the full result). Never pass `--strict` unless asked.
8. **Interpret.** Report the test, the statistic and p, the effect size with its CI, then the assumption checks and the cross-check verdict, in that order. If the result is **FLAGGED**:
   - cross-check disagrees → say the result cannot be trusted yet, show both numbers, and stop; a human must look.
   - an assumption failed → explain which, what it means for this test, and offer the pipeline's `fallback` (run it too if the user agrees; report both).
   - checks marked `?` are design properties: ask the user to confirm them.
9. **Never** change the test silently. The pipeline's own automatic switches (Fisher on sparse 2×2, negative binomial on overdispersion) are announced in `warnings`; repeat them.

## Spec skeleton

```json
{
  "test": "<test_id>",
  "design": {"goal": "...", "outcome_type": "...", "structure": "...", "predictors": "..."},
  "variables": {"outcome": "col", "group": "col"},
  "params": {},
  "alpha": 0.05, "alternative": "two-sided", "ci_level": 0.95,
  "decisions": {},
  "reasoning": ["answered <key>=<value> because ...", "ruled out <test> because ...", "columns inspected by header only"]
}
```

`design` keys are the question keys; `test` must be what the design resolves to, or a test on its fallback chain (then say why in `reasoning`).

## Reporting template

```
Test: <name> (<test_id>, status)
Why: <two sentences from reasoning>
Result: <statistic> = …, df = …, p = …; <effect size> = … [95% CI …]
Assumptions: <ok / failed / to confirm>
Cross-check (<library>): agrees / DISAGREES
Caveats: <warnings, fallback, what the user must confirm>
```

## When the taxonomy doesn't fit

Say so. Don't force a path. Common cases: more than one outcome type, a design with both pairing and clustering, time-varying covariates, non-inferiority margins. State what's missing and hand the choice to the user.
