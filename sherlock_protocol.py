# Utility functions for Talk2Table
from __future__ import annotations

import re
import unicodedata
import operator as _op
from typing import Any, Iterable, Callable

import pandas as pd
from rapidfuzz import process, fuzz


__all__ = [
    "_canon",
    "fuzzy_filter",
    "best_match",
    "list_distinct",
    "numeric_range",
    "safe_agg",
    "standardise_df",
    "standardise_column",
    # Sherlock helpers
    "equal_filter",
    "prefix_filter",
    "substring_filter",
    "regex_filter",
    "numeric_filter",
    "null_filter",
    "duplicate_filter",
    "row_any",
]


# ── string canonicaliser ───────────────────────────────────────────────

def _canon(text: str | Any) -> str:
    """Cheap, locale‑agnostic canonical form for fuzzy matching."""
    if text is None:
        return ""  # treat NaNs / None / etc. as empty string
    if not isinstance(text, str):
        text = str(text)
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", errors="ignore")
        .decode()
    )
    text = re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


# ── fuzzy helpers ──────────────────────────────────────────────────────

def fuzzy_filter(
    df: pd.DataFrame,
    column: str,
    query: str,
    *,
    cutoff: float = 0.8,
    top: int = 10,
) -> pd.DataFrame:
    """Return rows whose *column* is fuzzy‑similar to *query*."""
    if column not in df.columns:
        raise KeyError(f"{column!r} not found")

    canon = df[column].fillna("").astype(str).map(_canon)
    canon_lookup = dict(zip(canon, df[column]))

    # Search distinct canonical forms for speed
    uniques = df[column].dropna().astype(str).unique()
    canon_uniques = [_canon(u) for u in uniques]

    matches = process.extract(
        _canon(query),
        canon_uniques,
        scorer=fuzz.WRatio,
        score_cutoff=int(cutoff * 100),
        limit=top,
    )

    keep = [canon_lookup[m[0]] for m in matches]
    return df[df[column].isin(keep)].copy()


def best_match(
    df: pd.DataFrame,
    column: str,
    query: str,
    start: float = 0.9,
    floor: float = 0.5,
    step: float = 0.1,
    top: int = 3,
) -> tuple[pd.DataFrame, float]:
    """Find best match lowering cutoff until *floor* or result found."""
    cut = start
    while cut >= floor:
        sub = fuzzy_filter(df, column, query, cutoff=cut, top=top)
        if not sub.empty:
            return sub, cut
        cut -= step
    return df.iloc[0:0].copy(), 0.0


def list_distinct(
    df: pd.DataFrame,
    column: str,
    limit: int = 50,
    *,
    canonise: bool = False,
) -> list[str]:
    """List up to *limit* distinct values from *column* (canonicalised optional)."""
    if column not in df.columns:
        raise KeyError(f"{column!r} not in DataFrame columns")

    series = df[column].dropna().astype(str)
    if canonise:
        series = series.map(_canon)
    return series.unique().tolist()[:limit]


# ── numeric helpers ────────────────────────────────────────────────────

def numeric_range(
    df: pd.DataFrame,
    column: str,
) -> tuple[float, float]:
    """Return (min, max) of numeric column (after cleaning % etc.)."""
    nums = standardise_column(df, column, kind="numeric")
    return float(nums.min()), float(nums.max())


def safe_agg(
    df: pd.DataFrame,
    group_by: list[str] | None,
    metrics: dict[str, list[str]],
) -> pd.DataFrame:
    """Aggregate *df* safely, returning a new DataFrame."""
    if group_by:
        return df.groupby(group_by).agg(metrics).reset_index()
    res = df.agg(metrics)
    return res if isinstance(res, pd.DataFrame) else res.to_frame().T
    # return df.agg(metrics).to_frame().T


# ── universal normalisation helpers ──────────────────────────────────

def standardise_df(
    df: pd.DataFrame,
    *,
    strings: bool = True,
    numerics: bool = False,
    datetimes: bool = False,
) -> pd.DataFrame:
    """Return a *copy* whose values are harmonised so downstream comparisons work."""
    out = df.copy(deep=False)

    if strings:
        obj_cols = out.select_dtypes(include="object").columns
        for col in obj_cols:
            out[col] = out[col].astype(str).map(_canon)

    if numerics:
        obj_cols = out.select_dtypes(include="object").columns
        for col in obj_cols:
            cleaned = out[col].astype(str).str.replace(r"[^\d\.\-eE%]", "", regex=True)
            pct = cleaned.str.endswith("%")
            nums = pd.to_numeric(cleaned.str.rstrip("%"), errors="coerce")
            out[col] = nums.div(100).where(pct, nums)

    if datetimes:
        obj_cols = out.select_dtypes(include="object").columns
        for col in obj_cols:
            parsed = pd.to_datetime(out[col], errors="coerce", utc=True)
            if parsed.notna().sum():
                out[col] = parsed

    return out


def standardise_column(
    df: pd.DataFrame,
    column: str,
    *,
    kind: str = "string",
) -> pd.Series:
    """Clean a single column from *df* and return the result."""
    if column not in df.columns:
        raise KeyError(f"{column!r} missing")

    if kind == "string":
        return df[column].astype(str).map(_canon)

    if kind == "numeric":
        cleaned = df[column].astype(str).str.replace(r"[^\d\.\-eE%]", "", regex=True)
        pct = cleaned.str.endswith("%")
        nums = pd.to_numeric(cleaned.str.rstrip("%"), errors="coerce")
        return nums.div(100).where(pct, nums)

    if kind == "datetime":
        return pd.to_datetime(df[column], errors="coerce", utc=True)

    raise ValueError("kind must be 'string', 'numeric', or 'datetime'")


# ── Sherlock row‑filter helpers (levels 1‑10) ────────────────────────────

# NOTE: re & pandas already imported above

# ── internal utilities ─────────────────────────────────────────────────

def _require(df: pd.DataFrame, column: str) -> pd.Series:
    """Return column Series or raise descriptive KeyError."""
    if column not in df.columns:
        raise KeyError(f"{column!r} not in DataFrame columns: {list(df.columns)}")
    return df[column]


def _canon_series(s: pd.Series) -> pd.Series:
    """Canonicalise a Series of strings via existing _canon."""
    return s.astype(str).map(_canon)

# ── simple text predicates ─────────────────────────────────────────────

def equal_filter(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Exact equality comparison (strings are canonicalised)."""
    col = _require(df, column)
    if col.dtype == object:
        return df[_canon_series(col) == _canon(value)].copy()
    return df[col == value].copy()


def prefix_filter(df: pd.DataFrame, column: str, prefix: str, *, case: bool = False) -> pd.DataFrame:
    col = _require(df, column).astype(str)
    if not case:
        col, prefix = col.str.lower(), prefix.lower()
    return df[col.str.startswith(prefix, na=False)].copy()


def substring_filter(df: pd.DataFrame, column: str, text: str, *, case: bool = False) -> pd.DataFrame:
    col = _require(df, column).astype(str)
    if not case:
        col, text = col.str.lower(), text.lower()
    return df[col.str.contains(re.escape(text), na=False)].copy()


def regex_filter(df: pd.DataFrame, column: str, pattern: str, *, flags: int = re.I) -> pd.DataFrame:
    col = _require(df, column).astype(str)
    return df[col.str.contains(pattern, flags=flags, na=False)].copy()

# ── numeric / datetime predicates ─────────────────────────────────────

_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "==": _op.eq,  "eq": _op.eq,
    "!=": _op.ne,  "ne": _op.ne,
    ">":  _op.gt,  "gt": _op.gt,
    ">=": _op.ge,  "ge": _op.ge,
    "<":  _op.lt,  "lt": _op.lt,
    "<=": _op.le,  "le": _op.le,
}

def numeric_filter(
    df: pd.DataFrame,
    column: str,
    op: str,
    value: float | Iterable[float],
) -> pd.DataFrame:
    """
    Numeric/date comparisons that Just Work:
        op ∈ {'==','!=','>','>=','<','<=','between','range'}
        value = scalar for binary ops, (lo, hi) tuple for between/range
    """
    ser = standardise_column(df, column, kind="numeric")  # handles %, commas, etc.
    if op in {"between", "range"}:
        lo, hi = value
        mask = ser.between(lo, hi, inclusive="both")
    else:
        func = _OPS.get(op)
        if func is None:
            raise ValueError(f"Unsupported operator {op!r}")
        mask = func(ser, value)
    return df[mask.fillna(False)].copy()

# ── null / duplicate helpers ─────────────────────────────────────

def null_filter(df: pd.DataFrame, column: str, *, is_null: bool = True) -> pd.DataFrame:
    col = _require(df, column)
    mask = col.isna() if is_null else col.notna()
    return df[mask].copy()

def duplicate_filter(df: pd.DataFrame, column: str, *, canonise: bool = True) -> pd.DataFrame:
    col = _require(df, column).astype(str)
    if canonise:
        col = _canon_series(col)
    dup_idx = col[col.duplicated(keep=False)].index
    return df.loc[dup_idx].copy()

# ── combinator ───────────────────────────────────────────────────

def row_any(df: pd.DataFrame, *dfs: pd.DataFrame) -> pd.DataFrame:
    """Union of rows from multiple intermediate filters."""
    idx = pd.Index([])  # start empty
    for sub in dfs:
        idx = idx.union(sub.index)
    return df.loc[idx].copy()

# ── internal helpers ─────────────────────────────────────────────

def _canon_series(s: pd.Series) -> pd.Series:
    return s.map(_canon)
