"""The spec: the only thing the reasoning layer hands to the pipeline.

A spec names a test, binds dataset columns to the roles the test needs, and
carries the design answers that led to the test. The pipeline validates every
spec against ``taxonomy.json`` before running anything.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .taxonomy import Taxonomy, TaxonomyError, load_taxonomy

__all__ = ["Design", "Variables", "Spec", "SpecError", "Alternative"]

Alternative = Literal["two-sided", "less", "greater"]


class SpecError(ValueError):
    """The spec is inconsistent with the taxonomy or missing something the test needs."""


class Design(BaseModel):
    """Answers to the taxonomy questions. Field names are the question ``key``s."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    outcome_type: str | None = None
    structure: str | None = None
    predictors: str | None = None
    repeated_pattern: str | None = None
    aim: str | None = None
    survival_aim: str | None = None
    multivariate_aim: str | None = None
    variable_types: str | None = None
    task: str | None = None

    def answers(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def resolve(self, taxonomy: Taxonomy | None = None) -> str:
        tax = taxonomy or load_taxonomy()
        try:
            return tax.resolve(self.answers())
        except TaxonomyError as e:
            raise SpecError(str(e)) from e


class Variables(BaseModel):
    """Which dataset columns play which role. Only the roles the test uses need to be set.

    Long-format repeated data uses ``outcome`` + ``subject`` + ``within``.
    Wide-format repeated data uses ``columns`` (one column per condition / time / rater),
    or ``before`` / ``after`` for a single pair.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str | None = Field(None, description="Outcome / dependent variable column.")
    group: str | None = Field(None, description="Grouping factor column.")
    subject: str | None = Field(None, description="Subject / block id for repeated measures in long format.")
    within: str | None = Field(None, description="Condition / time column for repeated measures in long format.")
    columns: list[str] | None = Field(None, description="Wide format: one column per condition, time point or rater.")
    before: str | None = Field(None, description="Wide format paired data: first measurement.")
    after: str | None = Field(None, description="Wide format paired data: second measurement.")
    x: str | None = Field(None, description="First variable for a correlation / simple regression predictor.")
    y: str | None = Field(None, description="Second variable for a correlation / simple regression outcome.")
    row: str | None = Field(None, description="Row variable for a contingency table.")
    col: str | None = Field(None, description="Column variable for a contingency table.")
    time: str | None = Field(None, description="Follow-up time for survival data.")
    event: str | None = Field(None, description="Event indicator (1 = event, 0 = censored).")
    exposure: str | None = Field(None, description="Exposure / person-time for count outcomes.")
    cluster: str | None = Field(None, description="Cluster id (site, family) for cluster-robust inference.")
    strata: str | None = Field(None, description="Stratification variable (e.g. for a stratified Cox model).")
    predictors: list[str] = Field(default_factory=list, description="Right-hand-side columns for a regression.")
    categorical: list[str] = Field(default_factory=list, description="Which predictors to treat as categorical.")
    formula: str | None = Field(None, description="Explicit patsy formula; overrides outcome/predictors.")

    def set_roles(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v not in (None, [], "")]


class Spec(BaseModel):
    """What the skill hands to the pipeline."""

    model_config = ConfigDict(extra="forbid")

    test: str
    design: Design | None = None
    variables: Variables = Field(default_factory=Variables)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific constants, e.g. mu0, p0, expected, table, weights.",
    )
    alpha: float = 0.05
    alternative: Alternative = "two-sided"
    ci_level: float = 0.95
    decisions: dict[str, Any] = Field(
        default_factory=dict,
        description="Human modelling decisions. Required for every `judgment` test.",
    )
    reasoning: list[str] = Field(
        default_factory=list, description="The skill's reasoning: checks run, alternatives ruled out."
    )

    # Informational fields. Filled from the taxonomy by `complete()`; if supplied they must agree.
    status: Literal["ready", "helper", "judgment"] | None = None
    call: str | None = None
    assumption_checks: list[str] | None = None
    fallback: str | None = None
    cross_check: str | None = None
    resolved_test: str | None = Field(
        None,
        description="The test the design resolves to, when `test` is a fallback of it.",
    )

    @model_validator(mode="after")
    def _basic(self) -> "Spec":
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if not (0 < self.ci_level < 1):
            raise ValueError("ci_level must be in (0, 1)")
        return self

    # ------------------------------------------------------------------
    @classmethod
    def from_design(cls, design: Design | dict[str, str], taxonomy: Taxonomy | None = None, **kw: Any) -> "Spec":
        """Build a spec by resolving a design through the taxonomy."""
        d = design if isinstance(design, Design) else Design(**design)
        tax = taxonomy or load_taxonomy()
        test = d.resolve(tax)
        return cls(test=test, design=d, **kw).complete(tax)

    def complete(self, taxonomy: Taxonomy | None = None) -> "Spec":
        """Fill the informational fields from the taxonomy and validate consistency.

        Raises :class:`SpecError` if the test is unknown, the design resolves to a
        different test that is not an ancestor on the fallback chain, or a supplied
        informational field disagrees with the taxonomy.
        """
        tax = taxonomy or load_taxonomy()
        try:
            info = tax.test(self.test)
        except TaxonomyError as e:
            raise SpecError(str(e)) from e

        updates: dict[str, Any] = {}
        if self.design is not None:
            resolved = self.design.resolve(tax)
            if resolved != self.test:
                if self.test not in _fallback_chain(tax, resolved):
                    raise SpecError(
                        f"design resolves to {resolved!r} but spec names {self.test!r}, "
                        f"which is not on its fallback chain {_fallback_chain(tax, resolved)}"
                    )
                updates["resolved_test"] = resolved

        expected = {
            "status": info.status,
            "call": info.call,
            "assumption_checks": info.check_keys,
            "fallback": info.fallback.to if info.fallback and info.fallback.to else (
                info.fallback.technique if info.fallback else None
            ),
            "cross_check": info.xcall,
        }
        for k, v in expected.items():
            cur = getattr(self, k)
            if cur is not None and cur != v:
                raise SpecError(f"spec.{k}={cur!r} disagrees with taxonomy ({v!r}) for {self.test!r}")
            updates[k] = v
        return self.model_copy(update=updates)

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return self.model_dump_json(exclude_none=True, **kw)


def _fallback_chain(tax: Taxonomy, test_id: str) -> list[str]:
    chain: list[str] = []
    cur = tax.test(test_id)
    seen = {test_id}
    while cur.fallback and cur.fallback.to and cur.fallback.to not in seen:
        chain.append(cur.fallback.to)
        seen.add(cur.fallback.to)
        cur = tax.test(cur.fallback.to)
    return chain
