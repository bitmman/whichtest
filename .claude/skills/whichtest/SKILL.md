---
name: whichtest
description: Pick the right statistical test for a study design, write the spec, run it with assumption checks and an independent cross-check, and explain the result. Use when someone describes a study, dataset or analysis question ("which test should I use", "compare groups", "is X associated with Y", "analyse this trial") or asks to run or interpret a test with the whichtest pipeline.
---

# whichtest: from study design to a checked result

You are the reasoning layer. The pipeline (`whichtest` CLI / Python package) is deterministic and knows nothing about the study; your only output to it is a **spec**. Keep that contract: do not compute statistics yourself, do not bypass the pipeline, do not edit the taxonomy to make a case fit.

## Privacy rules (non-negotiable)

- Never read, print or paste raw data rows. Work from column names, dtypes, and summary counts only: `python -c "import pandas as pd; d = pd.read_csv('data.csv'); print(d.dtypes); print(d.shape)"`. Do not `head` or `cat` the file: a headerless file would print an observation. If the first line is not a header, ask the user for the column names instead. Say in your reasoning that columns were inspected by name and type only.
- The spec contains column names and design metadata. The pipeline's report contains statistics, counts, check verdicts and the user's decisions, and never an individual observation (assumption checks report how many values are outlying, not which). Those two things are all that may leave the machine, and only if the user chose a cloud model.

## Workflow

1. **Elicit the design.** Ask only what the taxonomy needs, in the taxonomy's order (`reference/questions.md`, or `whichtest tree`). The questions are: goal → outcome type → structure → predictors / aim. Stop asking once a leaf is reached. If the user already told you an answer, don't ask again. Record each answer and the reason you chose it.
2. **Resolve the test.** `whichtest resolve goal=... outcome_type=... structure=... predictors=...` prints the spec template with the taxonomy's status, checks, fallback and cross-check. `whichtest show <test_id>` gives the full entry, the variable roles it needs, and (for judgment tests) the decision keys.
3. **Rule out neighbours.** Name the one or two tests you did *not* pick and why (e.g. "paired, not independent, because the same patients were measured twice"). This goes into `reasoning`.
4. **Bind the columns.** Look at the column names and dtypes. Fill `variables` with the roles from `whichtest show`. Binary outcomes and truth labels are 0/1, or you name the positive level in `params.positive_label` (`binom_test`, `two_by_two`, `logistic`, `diag_binary`, `roc_auc`; a predicted-test column in `diag_binary` uses `params.test_positive_label`). Survival event columns are 1 = event, 0 = censored, except `competing`, which needs the distinct event codes kept as they are (`event_of_interest` picks one; do not collapse them to 1). Repeated data are either wide (`columns`) or long (`outcome` + `subject` + `within`); GEE with `ar1` or `unstructured` correlation also needs `time` = integer visit position.
5. **Judgment tests.** If the status is `judgment`, the pipeline will refuse to run until `decisions` is filled. Do not fill it yourself. `whichtest show <test_id>` lists every decision with its accepted values and when it is required (for `factor_sem`: `approach`, then `n_factors` + `rotation` or `model_syntax`). Present the options to the user with a plain recommendation and the trade-off, then record their choice exactly: booleans as JSON `true`/`false`, integers as numbers, choices as the listed spellings. The pipeline rejects look-alikes (`"false"`, `2.9`) rather than guessing.
6. **Write the spec** to a file and validate: `whichtest check spec.json`. It checks the taxonomy path, every required decision and its type, and the variable roles, without touching data; a nonzero exit prints what is missing. Column names and types are checked when you run.
7. **Run**: `whichtest run spec.json data.csv` (add `--json --out result.json` to keep the full result). Never pass `--strict` unless asked.
8. **Interpret.** Report the test, the statistic and p, the effect size with its CI, then the assumption checks and the cross-check verdict, in that order. Some results legitimately have no headline statistic (`lmm`, cumulative incidence): then the answer is the relevant coefficient or estimate with its CI, and you say so; never present a model's omnibus statistic as the test of the question (for an arm-by-time question in `lmm`, report the interaction coefficient). If the result is **FLAGGED**:
   - cross-check disagrees → say the result cannot be trusted yet, show both numbers, and stop; a human must look.
   - an assumption failed → explain which, what it means for this test, and offer the pipeline's `fallback` (run it too if the user agrees; report both).
   - checks marked `?` are design properties: ask the user to confirm them.
9. **Never** change the test silently. The pipeline's own automatic switches (Fisher on sparse 2×2, negative binomial on overdispersion) are announced in `warnings`; repeat them.

## Before you stop at a leaf

The questions reach a leaf quickly; three designs look like a leaf they are not. Ask about the estimand before accepting these:

- **Baseline and follow-up in two arms** ("same patients measured twice") is not `paired_t`. The question is the adjusted between-arm difference: `multiple_reg` with the baseline value and the arm as predictors (or `ancova`). A pooled paired test answers a different question.
- **Several visits in two arms, even with complete data,** is not a one-factor `rm_anova`: the arm-by-time term is the question. Keep it (`lmm` or GEE with the interaction in `formula`), and the random effects and REML are the user's decisions.
- **Deaths before relapse** when the question is "probability of relapse by time t" is `competing` with `estimand=cumulative_incidence`, not `km_logrank` on relapse with deaths censored, which overstates the probability. Cause-specific hazards are the choice when the question is etiological.

When a study mixes requirements the taxonomy keeps apart (pairing and clustering, several outcome types), say so and hand the choice to the user instead of forcing a path.

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
Result: <statistic> = …, df = …, p = …; <effect size> = … [95% CI …]   (or: <coefficient of interest> = … [95% CI …], p = … when the test has no headline statistic)
Decisions: <the user's choices, as echoed by the pipeline>  (judgment tests only)
Assumptions: <ok / failed / to confirm>
Cross-check (<library>): agrees / DISAGREES
Caveats: <warnings, fallback, what the user must confirm>
```

## When the taxonomy doesn't fit

Say so. Don't force a path. Common cases: more than one outcome type, a design with both pairing and clustering, time-varying covariates, non-inferiority margins. State what's missing and hand the choice to the user.
