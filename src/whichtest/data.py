"""Pull arrays out of a DataFrame according to the spec's variable bindings."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .spec import Spec, SpecError

__all__ = ["column", "groups", "paired", "blocks", "contingency_table", "need", "binary_labels"]


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


def paired(df: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Two aligned arrays (first, second) from wide (before/after) or long (subject/within/outcome) data."""
    v = spec.variables
    if v.before and v.after:
        a = column(df, v.before, "before")
        b = column(df, v.after, "after")
        labels = (v.before, v.after)
    elif v.columns and len(v.columns) == 2:
        a = column(df, v.columns[0], "columns[0]")
        b = column(df, v.columns[1], "columns[1]")
        labels = (v.columns[0], v.columns[1])
    elif v.outcome and v.subject and v.within:
        m = blocks(df, spec)
        if m.shape[1] != 2:
            raise SpecError(f"paired data needs exactly two levels of {v.within!r}; found {m.shape[1]}")
        a, b = m.iloc[:, 0].to_numpy(float), m.iloc[:, 1].to_numpy(float)
        labels = (str(m.columns[0]), str(m.columns[1]))
    else:
        raise SpecError(
            f"{spec.test!r} needs paired data: set variables.before/after, variables.columns (two), "
            "or variables.outcome + subject + within"
        )
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < len(a):
        a, b = a[keep], b[keep]
    return a, b, labels


def blocks(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Subjects × conditions matrix from wide (columns) or long (outcome/subject/within) data.

    Incomplete blocks are dropped and reported through the DataFrame's ``attrs['dropped']``.
    """
    v = spec.variables
    if v.columns:
        for c in v.columns:
            if c not in df.columns:
                raise SpecError(f"column {c!r} (variables.columns) is not in the data")
        m = df[v.columns].apply(pd.to_numeric, errors="coerce")
    elif v.outcome and v.subject and v.within:
        for c, r in ((v.outcome, "outcome"), (v.subject, "subject"), (v.within, "within")):
            if c not in df.columns:
                raise SpecError(f"column {c!r} (variables.{r}) is not in the data")
        if df.duplicated([v.subject, v.within]).any():
            raise SpecError(f"more than one row per ({v.subject}, {v.within}); repeated measures must be unique")
        m = df.pivot(index=v.subject, columns=v.within, values=v.outcome).apply(pd.to_numeric, errors="coerce")
    else:
        raise SpecError(
            f"{spec.test!r} needs repeated-measures data: set variables.columns (wide) "
            "or variables.outcome + subject + within (long)"
        )
    complete = m.dropna()
    complete.attrs["dropped"] = int(len(m) - len(complete))
    complete.attrs["n_total"] = int(len(m))
    return complete


def contingency_table(df: pd.DataFrame | None, spec: Spec) -> tuple[np.ndarray, list[Any], list[Any]]:
    """A contingency table from ``params['table']`` or a crosstab of variables.row × variables.col."""
    t = spec.params.get("table")
    if t is not None:
        arr = np.asarray(t, dtype=float)
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
    if numeric:
        try:
            pos_f = float(positive)
        except (TypeError, ValueError):
            raise SpecError(f"params.{param}={positive!r} is not a level of numeric column {name!r} ({levels})") from None
        if pos_f not in {float(v) for v in levels}:
            raise SpecError(f"params.{param}={positive!r} is not among the levels of {name!r}: {levels}")
        lab = (vals.astype(float) == pos_f).astype(int)
    else:
        if positive not in levels and str(positive) not in [str(v) for v in levels]:
            raise SpecError(f"params.{param}={positive!r} is not among the levels of {name!r}: {levels}")
        lab = (vals.astype(str) == str(positive)).astype(int)
    if len(levels) > 2:
        raise SpecError(f"{name!r} has {len(levels)} levels; a binary column was expected: {levels}")
    out = np.zeros(len(raw), dtype=int)
    out[valid] = lab
    return out, valid
