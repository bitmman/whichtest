"""Pull arrays out of a DataFrame according to the spec's variable bindings."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .spec import Spec, SpecError

__all__ = ["column", "groups", "paired", "blocks", "contingency_table", "need", "binary_labels", "validate_counts", "int_param", "positive_float_param", "float_param", "bool_param", "formula_columns"]

_REQUIRED = object()


def need(spec: Spec, *roles: str) -> None:
    """Raise a clear error unless every role in ``roles`` is bound.

    A role written ``a|b`` is satisfied by either binding.
    """
    v = spec.variables
    missing = []
    for r in roles:
        alts = r.split("|")
        if not any(getattr(v, a) not in (None, [], "") for a in alts):
            missing.append(r)
    if missing:
        raise SpecError(
            f"{spec.test!r} needs variables {missing} bound; set spec.variables "
            f"(currently: {spec.variables.set_roles() or 'nothing'})"
        )


def column(df: pd.DataFrame, name: str | None, role: str, numeric: bool = True) -> np.ndarray:
    if name is None:
        raise SpecError(f"variables.{role} is not set")
    if name not in df.columns:
        raise SpecError(f"column {name!r} (variables.{role}) is not in the data; columns: {list(df.columns)}")
    s = df[name]
    if numeric:
        s = pd.to_numeric(s, errors="coerce")
        arr = s.to_numpy(dtype=float)
        n_na = int(np.isnan(arr).sum())
        if n_na == len(arr):
            raise SpecError(f"column {name!r} has no numeric values")
        return arr
    return s.to_numpy()


def groups(df: pd.DataFrame, outcome: str | None, group: str | None, dropna: bool = True) -> dict[Any, np.ndarray]:
    """Outcome arrays keyed by group label, in order of first appearance."""
    y = column(df, outcome, "outcome")
    g = column(df, group, "group", numeric=False)
    out: dict[Any, np.ndarray] = {}
    for label in pd.unique(g):
        if pd.isna(label):
            continue
        vals = y[g == label]
        if dropna:
            vals = vals[~np.isnan(vals)]
        out[label] = vals
    if len(out) < 2:
        raise SpecError(f"group column {group!r} has fewer than two groups")
    return out


def paired(df: pd.DataFrame, spec: Spec, numeric: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Two aligned arrays (first, second) from wide (before/after) or long (subject/within/outcome) data.

    ``numeric=False`` keeps the values as labels (for nominal procedures); rows with a missing
    value in either column are dropped in both cases.
    """
    v = spec.variables
    if v.before and v.after:
        a = column(df, v.before, "before", numeric)
        b = column(df, v.after, "after", numeric)
        labels = (v.before, v.after)
    elif v.columns and len(v.columns) == 2:
        a = column(df, v.columns[0], "columns[0]", numeric)
        b = column(df, v.columns[1], "columns[1]", numeric)
        labels = (v.columns[0], v.columns[1])
    elif v.outcome and v.subject and v.within:
        m = blocks(df, spec, numeric)
        if m.shape[1] != 2:
            raise SpecError(f"paired data needs exactly two levels of {v.within!r}; found {m.shape[1]}")
        a, b = m.iloc[:, 0].to_numpy(float if numeric else object), m.iloc[:, 1].to_numpy(float if numeric else object)
        labels = (str(m.columns[0]), str(m.columns[1]))
    else:
        raise SpecError(
            f"{spec.test!r} needs paired data: set variables.before/after, variables.columns (two), "
            "or variables.outcome + subject + within"
        )
    keep = ~(pd.isna(a) | pd.isna(b))
    if keep.sum() < len(a):
        a, b = a[keep], b[keep]
    return a, b, labels


def blocks(df: pd.DataFrame, spec: Spec, numeric: bool = True) -> pd.DataFrame:
    """Subjects × conditions matrix from wide (columns) or long (outcome/subject/within) data.

    ``numeric=False`` keeps labels as they are (nominal ratings). Incomplete blocks are dropped
    and reported through the DataFrame's ``attrs['dropped']``.
    """
    v = spec.variables

    def conv(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(pd.to_numeric, errors="coerce") if numeric else frame.astype(object)

    if v.columns:
        for c in v.columns:
            if c not in df.columns:
                raise SpecError(f"column {c!r} (variables.columns) is not in the data")
        m = conv(df[v.columns])
    elif v.outcome and v.subject and v.within:
        for c, r in ((v.outcome, "outcome"), (v.subject, "subject"), (v.within, "within")):
            if c not in df.columns:
                raise SpecError(f"column {c!r} (variables.{r}) is not in the data")
        if df.duplicated([v.subject, v.within]).any():
            raise SpecError(f"more than one row per ({v.subject}, {v.within}); repeated measures must be unique")
        m = conv(df.pivot(index=v.subject, columns=v.within, values=v.outcome))
    else:
        raise SpecError(
            f"{spec.test!r} needs repeated-measures data: set variables.columns (wide) "
            "or variables.outcome + subject + within (long)"
        )
    complete = m.dropna()
    complete.attrs["dropped"] = int(len(m) - len(complete))
    complete.attrs["n_total"] = int(len(m))
    return complete


def validate_counts(values: Any, what: str, allow_zero: bool = True) -> np.ndarray:
    """Finite, non-negative, integer-valued counts, validated *before* any cast. Returns a float array."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise SpecError(f"{what} must be numeric counts") from None
    if not np.isfinite(arr).all():
        raise SpecError(f"{what} must be finite (no NaN/inf)")
    if (arr < 0).any() or (not allow_zero and (arr <= 0).any()):
        raise SpecError(f"{what} must be non-negative")
    if (arr != np.round(arr)).any():
        raise SpecError(f"{what} must be whole numbers (counts); got a fractional value")
    return arr


def int_param(params: dict, name: str, minimum: int = 0, default: Any = _REQUIRED) -> int:
    """An integer-valued, finite parameter ≥ ``minimum``; rejects 2.8, -0.5, inf and NaN before casting."""
    if name not in params or params[name] is None:
        if default is _REQUIRED:
            raise SpecError(f"params.{name} is required")
        return int(default)
    if isinstance(params[name], (bool, np.bool_, str)):
        raise SpecError(f"params.{name} must be an integer; got {params[name]!r}")
    arr = validate_counts([params[name]], f"params.{name}")
    v = int(arr[0])
    if v < minimum:
        raise SpecError(f"params.{name} must be ≥ {minimum}")
    return v


def positive_float_param(params: dict, name: str) -> float:
    if name not in params:
        raise SpecError(f"params.{name} is required")
    try:
        v = float(params[name])
    except (TypeError, ValueError):
        raise SpecError(f"params.{name} must be numeric") from None
    if not np.isfinite(v) or v <= 0:
        raise SpecError(f"params.{name} must be a positive finite number")
    return v


def float_param(params: dict, name: str, default: Any = _REQUIRED, lo: float | None = None, hi: float | None = None, open_interval: bool = False) -> float:
    """A finite numeric parameter, optionally within [lo, hi] (or (lo, hi) when ``open_interval``)."""
    if name not in params or params[name] is None:
        if default is _REQUIRED:
            raise SpecError(f"params.{name} is required")
        return float(default)
    v = params[name]
    if isinstance(v, (bool, np.bool_, str)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise SpecError(f"params.{name} must be a number; got {v!r}")
    f = float(v)
    if not np.isfinite(f):
        raise SpecError(f"params.{name} must be finite; got {v!r}")
    below = lo is not None and (f <= lo if open_interval else f < lo)
    above = hi is not None and (f >= hi if open_interval else f > hi)
    if below or above:
        br = ("(", ")") if open_interval else ("[", "]")
        raise SpecError(f"params.{name} must be in {br[0]}{lo}, {hi}{br[1]}; got {v!r}")
    return f


def bool_param(params: dict, name: str, default: bool) -> bool:
    """A JSON boolean; ``"false"``, ``0`` and other truthy/falsy stand-ins are rejected, not coerced."""
    if name not in params or params[name] is None:
        return bool(default)
    v = params[name]
    if not isinstance(v, (bool, np.bool_)):
        raise SpecError(f"params.{name} must be true or false; got {v!r}")
    return bool(v)


def contingency_table(df: pd.DataFrame | None, spec: Spec) -> tuple[np.ndarray, list[Any], list[Any]]:
    """A contingency table from ``params['table']`` or a crosstab of variables.row × variables.col."""
    t = spec.params.get("table")
    if t is not None:
        arr = validate_counts(t, "params.table")
        if arr.ndim != 2:
            raise SpecError("params.table must be a 2-D array of counts")
        return arr, list(range(arr.shape[0])), list(range(arr.shape[1]))
    v = spec.variables
    if df is None or not (v.row and v.col):
        raise SpecError(f"{spec.test!r} needs params.table or variables.row + variables.col")
    for c, r in ((v.row, "row"), (v.col, "col")):
        if c not in df.columns:
            raise SpecError(f"column {c!r} (variables.{r}) is not in the data")
    ct = pd.crosstab(df[v.row], df[v.col])
    return ct.to_numpy(dtype=float), list(ct.index), list(ct.columns)


def binary_labels(df: pd.DataFrame, name: str | None, role: str, positive: Any = None, param: str = "positive_label") -> tuple[np.ndarray, np.ndarray]:
    """0/1 labels and a validity mask for a binary column, comparing typed values, not strings.

    ``positive`` names the positive class. When it is ``None`` the column must be 0/1 (ints,
    floats or booleans); anything else must say which level is positive via ``params[param]``.
    Missing values are *not* negatives: they are ``False`` in the returned mask and the caller
    must filter every column with one joint mask.
    """
    raw = column(df, name, role, numeric=False)
    valid = ~pd.isna(raw)
    vals = raw[valid]
    levels = list(pd.unique(vals))
    numeric = all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in levels)
    if positive is None:
        if numeric and {float(v) for v in levels} <= {0.0, 1.0}:
            positive = 1
        else:
            raise SpecError(f"set params.{param}: {name!r} has levels {levels}")
    if len(levels) > 2:
        raise SpecError(f"{name!r} has {len(levels)} levels; a binary column was expected: {levels}")
    # The positive level need not be observed: a test that calls nobody positive is still a binary
    # test. It must be a valid level of the schema, though: a 0/1 column accepts 0 or 1, and a
    # labelled column with two observed levels must contain the declared positive.
    if numeric:
        try:
            pos_f = float(positive)
        except (TypeError, ValueError):
            raise SpecError(f"params.{param}={positive!r} is not a level of numeric column {name!r} ({levels})") from None
        observed = {float(v) for v in levels}
        if pos_f not in observed and not (observed <= {0.0, 1.0} and pos_f in (0.0, 1.0)):
            raise SpecError(f"params.{param}={positive!r} is not among the levels of {name!r}: {levels}")
        lab = (vals.astype(float) == pos_f).astype(int)
    else:
        observed_s = [str(v) for v in levels]
        if str(positive) not in observed_s and len(levels) >= 2:
            raise SpecError(f"params.{param}={positive!r} is not among the levels of {name!r}: {levels}")
        lab = (vals.astype(str) == str(positive)).astype(int)
    out = np.zeros(len(raw), dtype=int)
    out[valid] = lab
    return out, valid


def formula_columns(formula: str, df: pd.DataFrame) -> list[str]:
    """DataFrame columns referenced by a patsy formula, by identifier token (not substring)."""
    import re

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    return [c for c in df.columns if c in tokens]
