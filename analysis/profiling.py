"""Automated data profiling & cleaning suggestions. Pure Pandas — no LLM."""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[\d\-\+\(\)\s]{7,15}$")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    likely_pii: str = ""
    outlier_count: int = 0


@dataclass
class ProfileReport:
    row_count: int
    column_count: int
    duplicate_rows: int
    columns: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    completeness_score: float = 100.0
    consistency_score: float = 100.0
    validity_score: float = 100.0
    quality_score: float = 100.0


def profile_dataframe(df: pd.DataFrame) -> ProfileReport:
    columns = []
    suggestions = []

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = round(missing / len(df) * 100, 2) if len(df) else 0.0
        pii = _detect_pii(series)
        outliers = _count_outliers(series) if pd.api.types.is_numeric_dtype(series) else 0

        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                missing_count=missing,
                missing_pct=missing_pct,
                unique_count=int(series.nunique(dropna=True)),
                likely_pii=pii,
                outlier_count=outliers,
            )
        )

        if missing_pct > 50:
            suggestions.append(f"Column '{col}' is {missing_pct}% missing — consider dropping it.")
        elif missing_pct > 0:
            suggestions.append(f"Column '{col}' has {missing} missing values — consider imputing or dropping rows.")
        if pii:
            suggestions.append(f"Column '{col}' looks like {pii} — consider masking before analysis.")
        if outliers > 0:
            suggestions.append(f"Column '{col}' has {outliers} statistical outliers (IQR method).")

    dup_count = int(df.duplicated().sum())
    if dup_count:
        suggestions.append(f"Found {dup_count} duplicate rows — consider dropping them.")

    completeness, consistency, validity, quality = _compute_quality_score(df, columns, dup_count)

    return ProfileReport(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=dup_count,
        columns=columns,
        suggestions=suggestions,
        completeness_score=completeness,
        consistency_score=consistency,
        validity_score=validity,
        quality_score=quality,
    )


def _compute_quality_score(df: pd.DataFrame, columns: list, dup_count: int) -> tuple:
    """Composite data-quality score (0-100) from completeness, consistency,
    and validity — a single deterministic number, no LLM judgment involved."""
    total_cells = len(df) * len(df.columns)
    total_missing = sum(c.missing_count for c in columns)
    completeness = 100.0 * (1 - total_missing / total_cells) if total_cells else 100.0

    consistency = 100.0 * (1 - dup_count / len(df)) if len(df) else 100.0

    numeric_cells = sum(len(df) for c in columns if pd.api.types.is_numeric_dtype(df[c.name]))
    total_outliers = sum(c.outlier_count for c in columns)
    validity = 100.0 * (1 - total_outliers / numeric_cells) if numeric_cells else 100.0

    quality = 0.4 * completeness + 0.3 * consistency + 0.3 * validity
    return round(completeness, 1), round(consistency, 1), round(validity, 1), round(quality, 1)


def _detect_pii(series: pd.Series) -> str:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return ""
    if sample.apply(lambda x: bool(EMAIL_RE.match(x))).mean() > 0.7:
        return "email"
    if series.name and re.search(r"phone|mobile|tel", str(series.name), re.IGNORECASE):
        return "phone"
    if series.name and re.search(r"ssn|national.?id|passport", str(series.name), re.IGNORECASE):
        return "national_id"
    return ""


def _count_outliers(series: pd.Series) -> int:
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def apply_cleaning(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    """Apply user-approved cleaning actions. actions: {"drop_duplicates": bool,
    "drop_columns": [...], "impute": {"col": "mean|median|mode|drop_rows"}, "mask_pii": [...]}"""
    out = df.copy()
    if actions.get("drop_duplicates"):
        out = out.drop_duplicates()
    for col in actions.get("drop_columns", []):
        if col in out.columns:
            out = out.drop(columns=[col])
    for col, method in actions.get("impute", {}).items():
        if col not in out.columns:
            continue
        if method == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif method == "median":
            out[col] = out[col].fillna(out[col].median())
        elif method == "mode":
            mode = out[col].mode()
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
        elif method == "drop_rows":
            out = out.dropna(subset=[col])
    for col in actions.get("mask_pii", []):
        if col in out.columns:
            out[col] = out[col].astype(str).apply(lambda x: x[:2] + "***" if len(x) > 2 else "***")
    return out
