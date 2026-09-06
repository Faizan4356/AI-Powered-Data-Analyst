"""Automated EDA: distributions, correlations, missingness, categorical breakdowns."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EDAReport:
    numeric_summary: pd.DataFrame
    categorical_summary: dict
    correlation_matrix: pd.DataFrame
    missingness: pd.DataFrame
    narrative: list = field(default_factory=list)


def run_eda(df: pd.DataFrame) -> EDAReport:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    numeric_summary = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = df[col].value_counts(dropna=False).head(10)

    correlation_matrix = df[numeric_cols].corr() if len(numeric_cols) > 1 else pd.DataFrame()

    missingness = df.isna().sum().to_frame("missing_count")
    missingness["missing_pct"] = (missingness["missing_count"] / len(df) * 100).round(2) if len(df) else 0.0

    narrative = _build_narrative(df, numeric_summary, correlation_matrix, missingness)

    return EDAReport(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        correlation_matrix=correlation_matrix,
        missingness=missingness,
        narrative=narrative,
    )


def _build_narrative(df, numeric_summary, corr, missingness) -> list:
    notes = [f"Dataset has {len(df)} rows and {len(df.columns)} columns."]

    high_missing = missingness[missingness["missing_pct"] > 20]
    if not high_missing.empty:
        notes.append(f"{len(high_missing)} column(s) have more than 20% missing values: {list(high_missing.index)}.")

    if not corr.empty:
        mask = np.eye(len(corr), dtype=bool)
        pairs = corr.mask(mask)
        strong = pairs[(pairs.abs() > 0.7)].stack()
        if not strong.empty:
            top = strong.abs().sort_values(ascending=False).head(3)
            for (a, b), _ in top.items():
                notes.append(f"Strong correlation between '{a}' and '{b}' ({corr.loc[a, b]:.2f}).")

    return notes
