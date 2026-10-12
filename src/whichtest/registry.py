"""Registry mapping ``test_id`` → implementation, and the ``run`` entry point."""
from __future__ import annotations

import importlib

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .result import TestResult
from .spec import Spec, SpecError
from .taxonomy import Taxonomy, load_taxonomy

__all__ = ["procedure", "run", "available", "preflight", "validate_decisions", "Decision", "Procedure", "PROCEDURES"]

ProcedureFn = Callable[[pd.DataFrame | None, Spec], TestResult]

_MISSING = (None, "", [])


@dataclass(frozen=True)
class Decision:
    """One human decision a judgment test needs, with its type and domain.

    ``kind``: ``choice`` (one of ``choices``, case-insensitive, after ``normalize``), ``bool`` (a JSON
    boolean; ``"false"`` and ``0`` are rejected because they are not a human choice), ``int`` (an
    integer value, optionally restricted to ``choices`` or ``>= minimum``; ``2.9`` is rejected rather
    than truncated), ``str`` (non-empty text; formulas and lavaan syntax are checked against the data
    inside the procedure), ``code`` (an event code; the procedure validates it against the data).
    ``when`` = ``(key, value)`` makes the decision required only when another decision has that value.
    """

    key: str
    doc: str
    kind: str = "choice"
    choices: tuple[Any, ...] = ()
    minimum: int | None = None
    when: tuple[str, str] | None = None
    normalize: Callable[[str], str] | None = None

    def domain(self) -> str:
        """The accepted values in words, plus the condition that makes the decision required."""
        if self.kind == "bool":
            dom = "true/false"
        elif self.choices:
            dom = " | ".join(str(c) for c in self.choices)
        elif self.kind == "int":
            dom = "integer" + (f" ≥ {self.minimum}" if self.minimum is not None else "")
        elif self.kind == "str":
            dom = "text"
        else:
            dom = "event code"
        return dom + (f"; required when {self.when[0]}={self.when[1]}" if self.when else "")

    def describe(self) -> str:
        return f"{self.key} ({self.domain()}): {self.doc}"

    def applies(self, decisions: dict[str, Any]) -> bool:
        if self.when is None:
            return True
        v = decisions.get(self.when[0])
        return isinstance(v, str) and v.strip().lower() == self.when[1]

    def check(self, value: Any) -> None:
        """Raise ``SpecError`` unless ``value`` is in this decision's domain. Never coerces."""
        name = f"decisions.{self.key}"
        if self.kind == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise SpecError(f"{name} must be a JSON boolean (true/false); got {value!r}")
            return
        if self.kind == "int":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise SpecError(f"{name} must be an integer; got {value!r}")
            f = float(value)
            if not np.isfinite(f) or f != round(f):
                raise SpecError(f"{name} must be an integer; got {value!r}")
            if self.choices and int(f) not in self.choices:
                raise SpecError(f"{name} must be one of {list(self.choices)}; got {value!r}")
            if self.minimum is not None and f < self.minimum:
                raise SpecError(f"{name} must be ≥ {self.minimum}; got {value!r}")
            return
        if self.kind in ("choice", "str"):
            if not isinstance(value, str) or not value.strip():
                raise SpecError(f"{name} must be non-empty text; got {value!r}")
            if self.kind == "choice":
                key = value.strip().lower()
                if self.normalize is not None:
                    key = self.normalize(key)
                if key not in {str(c).lower() for c in self.choices}:
                    raise SpecError(f"{name} must be one of {list(self.choices)}; got {value!r}")
            return
        if self.kind == "code":
            if isinstance(value, (bool, np.bool_)) or value in _MISSING:
                raise SpecError(f"{name} must be an event code (number or label); got {value!r}")
            return
        raise RuntimeError(f"unknown decision kind {self.kind!r}")  # pragma: no cover


@dataclass
class Procedure:
    test_id: str
    fn: ProcedureFn
    requires: tuple[str, ...] = ()
    decisions: tuple[Decision, ...] = field(default_factory=tuple)  # judgment tests: the human decisions
    data_optional: bool = False  # tests that can run from params alone (e.g. a count table)
    directional: bool = False  # honours spec.alternative in primary *and* cross-check; else only two-sided is accepted

    @property
    def decision_keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.decisions)


PROCEDURES: dict[str, Procedure] = {}

_MODULES = (
    "whichtest.procedures.parametric",
    "whichtest.procedures.nonparametric",
    "whichtest.procedures.categorical",
    "whichtest.procedures.regression",
    "whichtest.procedures.count",
    "whichtest.procedures.survival",
    "whichtest.procedures.correlation",
    "whichtest.procedures.agreement",
    "whichtest.procedures.multivariate",
    "whichtest.procedures.mixed",
)
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    for m in _MODULES:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:  # a module that isn't written yet
            if e.name != m:
                raise
    _loaded = True


def procedure(test_id: str, *, requires: tuple[str, ...] = (), decisions: tuple[Decision, ...] = (), data_optional: bool = False, directional: bool = False):
    """Register ``fn`` as the implementation of ``test_id``.

    ``requires`` lists variable roles that must be bound (``"a|b"`` = either).
    ``decisions`` lists the :class:`Decision` objects ``spec.decisions`` must satisfy (judgment tests).
    ``directional`` says the procedure applies ``spec.alternative`` consistently; otherwise
    ``run()`` rejects a one-sided alternative instead of silently reporting two-sided results.
    """
    if any(isinstance(d, str) for d in decisions):
        raise TypeError("decisions must be Decision objects, not bare keys")

    def deco(fn: ProcedureFn) -> ProcedureFn:
        if test_id in PROCEDURES:
            raise RuntimeError(f"{test_id!r} registered twice")
        PROCEDURES[test_id] = Procedure(test_id, fn, requires, tuple(decisions), data_optional, directional)
        return fn

    return deco


def validate_decisions(proc: Procedure, decisions: dict[str, Any]) -> None:
    """Every applicable decision is present and in its domain; unknown keys are rejected.

    Checks only; the dictionary is never coerced or modified, so the decisions echoed in the
    result are exactly what the human wrote.
    """
    known = {d.key for d in proc.decisions}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise SpecError(f"{proc.test_id!r} does not take decisions {unknown}; it takes {sorted(known)}")
    missing = []
    for d in proc.decisions:
        present = decisions.get(d.key) not in _MISSING
        if present:
            d.check(decisions[d.key])
        elif d.applies(decisions):
            missing.append(d)
    if missing:
        raise SpecError(
            f"{proc.test_id!r} is a judgment test; it will not run until spec.decisions sets "
            f"{[d.key for d in missing]}: " + "; ".join(d.describe() for d in missing)
        )


def preflight(spec: Spec | dict[str, Any], taxonomy: Taxonomy | None = None) -> tuple[Spec, Procedure]:
    """Everything ``run()`` checks before it touches data: taxonomy consistency, an implementation,
    complete and well-typed decisions, bound variable roles, and a supported alternative."""
    _load()
    tax = taxonomy or load_taxonomy()
    if not isinstance(spec, Spec):
        spec = Spec(**spec)
    spec = spec.complete(tax)
    info = tax.test(spec.test)
    proc = PROCEDURES.get(spec.test)
    if proc is None:
        raise NotImplementedError(
            f"{spec.test!r} ({info.name}, status={info.status}) has no implementation yet. "
            f"Taxonomy suggests: {info.call}"
        )
    if info.status == "judgment" or proc.decisions or spec.decisions:
        validate_decisions(proc, spec.decisions)
    if spec.alternative != "two-sided" and not proc.directional:
        raise SpecError(
            f"{spec.test!r} has no one-sided version; it reports a two-sided p-value. "
            f"Set alternative='two-sided' (got {spec.alternative!r})"
        )
    if proc.requires:
        from .data import need

        need(spec, *proc.requires)
    return spec, proc


def _formula_errors() -> tuple[type[BaseException], ...]:
    try:
        from patsy import PatsyError
    except ImportError:  # pragma: no cover
        return ()
    return (PatsyError,)


def available() -> list[str]:
    _load()
    return sorted(PROCEDURES)


def run(spec: Spec | dict[str, Any], data: pd.DataFrame | None = None, taxonomy: Taxonomy | None = None) -> TestResult:
    """Validate ``spec`` against the taxonomy, run the test, cross-check it, and return the result."""
    tax = taxonomy or load_taxonomy()
    spec, proc = preflight(spec, tax)
    info = tax.test(spec.test)
    if data is None and not proc.data_optional:
        raise SpecError(f"{spec.test!r} needs a DataFrame")
    if data is not None and not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    try:
        result = proc.fn(data, spec)
    except _formula_errors() as e:
        # a formula that names a missing column or cannot be parsed is a spec problem, not a crash
        raise SpecError(f"{spec.test!r}: formula error: {e}") from None
    result.test = spec.test
    result.name = result.name or info.name
    result.status = info.status
    result.alpha = spec.alpha
    if result.cross_check is None:
        from .result import CrossCheck

        result.cross_check = CrossCheck(
            method=info.xcall,
            why=info.cross_why,
            kind="exact",
            agrees=False,
            detail="NOT RUN: the cross-check library is not installed (pip install 'whichtest[cross]'); the result is unverified",
        )
        result.warnings.append("cross-check not run: optional dependency missing; result flagged as unverified")
    if info.status == "judgment":
        # every judgment result carries the human decisions it was run with
        result.extra["decisions"] = dict(spec.decisions)
    if result.fallback is None and info.fallback and result.failed_checks:
        result.fallback = {
            "to": info.fallback.to,
            "technique": info.fallback.technique,
            "text": info.fallback.text,
            "because": [c.key for c in result.failed_checks],
        }
    return result
