"""Structured results: what every procedure returns."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np

__all__ = ["CheckResult", "Estimate", "Comparison", "CrossCheck", "TestResult", "compare"]


def _clean(v: Any) -> Any:
    """Make values JSON-friendly."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "to_dict") and not isinstance(v, TestResult):
        try:
            return _clean(v.to_dict())
        except Exception:  # pragma: no cover
            return str(v)
    return v


@dataclass
class CheckResult:
    """Outcome of one assumption check.

    ``passed`` is ``None`` when the check cannot be decided from the data
    (e.g. independence, which is a design property): the text then tells the
    human what to confirm.
    """

    key: str
    text: str
    passed: bool | None = None
    statistic: float | None = None
    p: float | None = None
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.passed is False


@dataclass
class Estimate:
    name: str
    value: float | None
    ci: tuple[float, float] | None = None
    ci_level: float | None = None
    note: str = ""


@dataclass
class Comparison:
    quantity: str
    primary: float | None
    secondary: float | None
    agrees: bool
    rule: str


@dataclass
class CrossCheck:
    method: str
    why: str
    kind: Literal["exact", "approximate", "direction"]
    agrees: bool
    comparisons: list[Comparison] = field(default_factory=list)
    detail: str = ""


@dataclass
class TestResult:
    test: str
    name: str = ""
    status: str = ""
    n: int | dict[str, int] | None = None
    statistic_name: str = ""
    statistic: float | None = None
    p: float | None = None
    df: float | tuple[float, float] | None = None
    alpha: float = 0.05
    estimates: list[Estimate] = field(default_factory=list)
    effect_size: Estimate | None = None
    checks: list[CheckResult] = field(default_factory=list)
    cross_check: CrossCheck | None = None
    fallback: dict[str, Any] | None = None
    post_hoc: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def significant(self) -> bool | None:
        """None when there is no p-value or inference failed (NaN p)."""
        if self.p is None or not math.isfinite(self.p):
            return None
        return bool(self.p < self.alpha)

    @property
    def inference_failed(self) -> bool:
        """A p-value or statistic that came back NaN/inf: the test ran but produced nothing usable."""
        return any(v is not None and isinstance(v, float) and not math.isfinite(v) for v in (self.p, self.statistic))

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.failed]

    @property
    def flagged(self) -> bool:
        """True when a human should look: no usable cross-check, cross-check disagreement, failed
        assumptions, or inference that produced non-finite numbers."""
        return (
            self.cross_check is None
            or not self.cross_check.agrees
            or bool(self.failed_checks)
            or self.inference_failed
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["significant"] = self.significant
        d["flagged"] = self.flagged
        return _clean(d)

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kw)

    def summary(self) -> str:
        lines = [f"{self.name} [{self.test}, {self.status}]"]
        if self.statistic is not None:
            dfs = "" if self.df is None else f", df = {_fmt(self.df)}"
            lines.append(f"  {self.statistic_name or 'statistic'} = {_fmt(self.statistic)}{dfs}, p = {_fmt_p(self.p)}")
        elif self.p is not None:
            lines.append(f"  p = {_fmt_p(self.p)}")
        if self.effect_size:
            lines.append("  effect size: " + _fmt_est(self.effect_size))
        for e in self.estimates:
            lines.append("  " + _fmt_est(e))
        if self.extra.get("decisions"):
            lines.append("  decisions (set by a human): " + ", ".join(f"{k}={v!r}" for k, v in self.extra["decisions"].items()))
        if self.checks:
            lines.append("  assumption checks:")
            for c in self.checks:
                mark = {True: "ok  ", False: "FAIL", None: "?   "}[c.passed]
                extra = f" (p = {_fmt_p(c.p)})" if c.p is not None else ""
                lines.append(f"    [{mark}] {c.key}: {c.text}{extra}{' — ' + c.detail if c.detail else ''}")
        if self.inference_failed:
            lines.append("  inference FAILED: statistic or p-value is not finite")
        if self.cross_check is None:
            lines.append("  cross-check: NOT RUN")
        else:
            cc = self.cross_check
            lines.append(f"  cross-check ({cc.kind}) via {cc.method}: {'AGREES' if cc.agrees else 'DISAGREES'}")
            for cmp in cc.comparisons:
                if not cmp.agrees:
                    lines.append(f"    {cmp.quantity}: {_fmt(cmp.primary)} vs {_fmt(cmp.secondary)} ({cmp.rule})")
            if cc.detail:
                lines.append(f"    {cc.detail}")
        if self.fallback:
            because = self.fallback.get("because")
            lines.append(f"  fallback suggested: {self.fallback.get('to') or self.fallback.get('technique')} — {self.fallback.get('text', '')}" + (f" (because: {', '.join(because)})" if because else ""))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        if self.flagged:
            lines.append("  >> FLAGGED for human review")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "NA"
    if math.isinf(f):
        return "-inf" if f < 0 else "inf"
    if f == int(f) and abs(f) < 1e6:
        return str(int(f))
    return f"{f:.4g}"


def _fmt_p(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "NA"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _fmt_est(e: Estimate) -> str:
    s = f"{e.name} = {_fmt(e.value)}"
    if e.ci is not None:
        lvl = f"{int(round((e.ci_level or 0.95) * 100))}%"
        s += f" [{lvl} CI {_fmt(e.ci[0])}, {_fmt(e.ci[1])}]"
    if e.note:
        s += f" ({e.note})"
    return s


# ----------------------------------------------------------------------
def compare(
    method: str,
    why: str,
    primary: dict[str, float | None],
    secondary: dict[str, float | None],
    *,
    kind: Literal["exact", "approximate", "direction"] = "exact",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    alpha: float = 0.05,
    p_tol: float = 0.02,
    detail: str = "",
) -> CrossCheck:
    """Compare matching quantities from two implementations.

    * ``exact``: every quantity must match within ``rtol``/``atol``.
    * ``approximate``: p-values must lead to the same decision at ``alpha`` or differ by
      less than ``p_tol``; other quantities must match within ``rtol``/``atol``.
    * ``direction``: quantities named ``direction`` must have the same sign; p-values are
      reported but not compared (one method may simply have less power).
    """
    comps: list[Comparison] = []
    for q, a in primary.items():
        b = secondary.get(q)
        if a is None or b is None:
            comps.append(Comparison(q, a, b, False, "missing value"))
            continue
        a, b = float(a), float(b)
        if not (math.isfinite(a) and math.isfinite(b)):
            comps.append(Comparison(q, a, b, False, "non-finite value"))
            continue
        if kind == "direction" and q.startswith("direction"):
            ok = bool(np.sign(a) == np.sign(b))
            rule = "same sign (zero counts as its own sign)"
        elif kind == "direction" and q.startswith("p"):
            ok, rule = True, "reported only"
        elif kind == "approximate" and q.startswith("p"):
            ok = ((a < alpha) == (b < alpha)) or abs(a - b) < p_tol
            rule = f"same decision at alpha={alpha} or |Δp| < {p_tol}"
        else:
            ok = bool(np.isclose(a, b, rtol=rtol, atol=atol))
            rule = f"rtol={rtol:g}, atol={atol:g}"
        comps.append(Comparison(q, a, b, ok, rule))
    return CrossCheck(method=method, why=why, kind=kind, agrees=all(c.agrees for c in comps), comparisons=comps, detail=detail)
