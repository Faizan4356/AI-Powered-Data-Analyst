"""Root-cause / driver analysis: decompose a metric's change between two
periods by a dimension (e.g. "revenue dropped mainly due to Region X, -18%").

Pure Pandas — sums the metric per dimension value in each period, then
attributes each value's share of the total change. No LLM involvement in
the numbers; the LLM (if used downstream) only narrates this already-computed
breakdown.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class DriverContribution:
    dimension_value: str
    value_a: float
    value_b: float
    change: float
    pct_change: float
    pct_of_total_change: float


@dataclass
class RootCauseResult:
    total_a: float
    total_b: float
    total_change: float
    total_pct_change: float
    drivers: list  # DriverContribution, sorted by |contribution| descending
    summary: str


def decompose_change(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    dimension_col: str,
    period_a_start,
    period_a_end,
    period_b_start,
    period_b_end,
) -> RootCauseResult:
    dates = pd.to_datetime(df[date_col])
    a_mask = (dates >= pd.Timestamp(period_a_start)) & (dates <= pd.Timestamp(period_a_end))
    b_mask = (dates >= pd.Timestamp(period_b_start)) & (dates <= pd.Timestamp(period_b_end))

    period_a = df[a_mask]
    period_b = df[b_mask]

    if period_a.empty or period_b.empty:
        raise ValueError("One or both periods contain no rows")

    a_by_dim = period_a.groupby(dimension_col)[metric_col].sum()
    b_by_dim = period_b.groupby(dimension_col)[metric_col].sum()

    all_dims = sorted(set(a_by_dim.index) | set(b_by_dim.index), key=str)

    total_a = float(period_a[metric_col].sum())
    total_b = float(period_b[metric_col].sum())
    total_change = total_b - total_a
    total_pct_change = (total_change / total_a * 100) if total_a else float("inf")

    drivers = []
    for dim in all_dims:
        val_a = float(a_by_dim.get(dim, 0.0))
        val_b = float(b_by_dim.get(dim, 0.0))
        change = val_b - val_a
        pct_change = (change / val_a * 100) if val_a else float("inf")
        pct_of_total = (change / total_change * 100) if total_change else 0.0
        drivers.append(
            DriverContribution(
                dimension_value=str(dim),
                value_a=round(val_a, 4),
                value_b=round(val_b, 4),
                change=round(change, 4),
                pct_change=round(pct_change, 2),
                pct_of_total_change=round(pct_of_total, 2),
            )
        )

    drivers.sort(key=lambda d: abs(d.change), reverse=True)

    summary = _build_summary(dimension_col, total_a, total_b, total_change, total_pct_change, drivers)

    return RootCauseResult(
        total_a=round(total_a, 4),
        total_b=round(total_b, 4),
        total_change=round(total_change, 4),
        total_pct_change=round(total_pct_change, 2),
        drivers=drivers,
        summary=summary,
    )


def _build_summary(dimension_col, total_a, total_b, total_change, total_pct_change, drivers) -> str:
    direction = "increased" if total_change >= 0 else "decreased"
    lines = [f"The metric {direction} from {total_a:.2f} to {total_b:.2f} ({total_pct_change:+.1f}%)."]

    if drivers:
        top = drivers[0]
        top_direction = "grew" if top.change >= 0 else "dropped"
        lines.append(
            f"'{top.dimension_value}' ({dimension_col}) {top_direction} the most, from {top.value_a:.2f} to "
            f"{top.value_b:.2f} ({top.pct_change:+.1f}%), accounting for {top.pct_of_total_change:.1f}% of the total change."
        )
    return " ".join(lines)
