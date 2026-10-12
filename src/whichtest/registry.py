"""Registry mapping ``test_id`` → implementation, and the ``run`` entry point."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .result import TestResult
from .spec import Spec, SpecError
from .taxonomy import Taxonomy, load_taxonomy

__all__ = ["procedure", "run", "available", "Procedure", "PROCEDURES"]

ProcedureFn = Callable[[pd.DataFrame | None, Spec], TestResult]


@dataclass
class Procedure:
    test_id: str
    fn: ProcedureFn
    requires: tuple[str, ...] = ()
    decisions: tuple[str, ...] = field(default_factory=tuple)  # judgment tests: required decision keys
    data_optional: bool = False  # tests that can run from params alone (e.g. a count table)


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


def procedure(test_id: str, *, requires: tuple[str, ...] = (), decisions: tuple[str, ...] = (), data_optional: bool = False):
    """Register ``fn`` as the implementation of ``test_id``.

    ``requires`` lists variable roles that must be bound (``"a|b"`` = either).
    ``decisions`` lists the keys ``spec.decisions`` must contain (judgment tests).
    """

    def deco(fn: ProcedureFn) -> ProcedureFn:
        if test_id in PROCEDURES:
            raise RuntimeError(f"{test_id!r} registered twice")
        PROCEDURES[test_id] = Procedure(test_id, fn, requires, decisions, data_optional)
        return fn

    return deco


def available() -> list[str]:
    _load()
    return sorted(PROCEDURES)


def run(spec: Spec | dict[str, Any], data: pd.DataFrame | None = None, taxonomy: Taxonomy | None = None) -> TestResult:
    """Validate ``spec`` against the taxonomy, run the test, cross-check it, and return the result."""
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
    if info.status == "judgment":
        missing = [k for k in proc.decisions if spec.decisions.get(k) in (None, "", [])]
        if missing:
            raise SpecError(
                f"{spec.test!r} is a judgment test; it will not run until spec.decisions sets {missing}"
            )
    if data is None and not proc.data_optional:
        raise SpecError(f"{spec.test!r} needs a DataFrame")
    if data is not None and not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if proc.requires:
        from .data import need

        need(spec, *proc.requires)

    result = proc.fn(data, spec)
    result.test = spec.test
    result.name = result.name or info.name
    result.status = info.status
    result.alpha = spec.alpha
    if result.fallback is None and info.fallback and result.failed_checks:
        result.fallback = {
            "to": info.fallback.to,
            "technique": info.fallback.technique,
            "text": info.fallback.text,
            "because": [c.key for c in result.failed_checks],
        }
    return result
