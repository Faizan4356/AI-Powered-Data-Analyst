"""Auto-chart-type selection based on the shape of an already-computed result.

Deterministic, not LLM-driven: the LLM never sees the result table for this
decision either, since which chart best fits a table's shape is a plain
structural fact (row/column counts, dtypes, cardinality), not something that
benefits from — or should risk — a model's judgment.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

MAX_CATEGORIES_FOR_BAR = 30


@dataclass
class ChartSpec:
    chart_type: str  # "bar" | "line" | "scatter" | "heatmap" | "histogram" | "table"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    reason: str = ""


def choose_chart(df: pd.DataFrame) -> ChartSpec:
    if df is None or df.empty or len(df.columns) == 0:
        return ChartSpec(chart_type="table", reason="Empty result")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or _looks_like_date(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    if len(df) == 1:
        return ChartSpec(chart_type="table", reason="Single-row result — a table/metric is clearer than a chart")

    if datetime_cols and numeric_cols:
        return ChartSpec(
            chart_type="line", x=datetime_cols[0], y=numeric_cols[0], reason="Time column + numeric column detected"
        )

    if len(categorical_cols) == 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        if df[cat_col].nunique() <= MAX_CATEGORIES_FOR_BAR:
            color = categorical_cols[1] if len(categorical_cols) > 1 else None
            return ChartSpec(
                chart_type="bar", x=cat_col, y=numeric_cols[0], color=color, reason="One low-cardinality category + numeric value"
            )
        return ChartSpec(chart_type="table", reason=f"Category column has too many distinct values ({df[cat_col].nunique()})")

    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        return ChartSpec(
            chart_type="heatmap", x=categorical_cols[0], y=categorical_cols[1], color=numeric_cols[0],
            reason="Two category dimensions + a numeric value — heatmap shows the cross-tab clearly",
        )

    if len(numeric_cols) >= 2:
        return ChartSpec(
            chart_type="scatter", x=numeric_cols[0], y=numeric_cols[1], reason="Two or more numeric columns — relationship view"
        )

    if len(numeric_cols) == 1 and not categorical_cols:
        return ChartSpec(chart_type="histogram", x=numeric_cols[0], reason="Single numeric column across many rows — distribution view")

    return ChartSpec(chart_type="table", reason="No chart-friendly shape detected")


def _looks_like_date(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    name = str(series.name).lower()
    return "date" in name or "time" in name
