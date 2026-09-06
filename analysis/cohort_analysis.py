"""Cohort analysis: retention over time by first-activity cohort. Pure Pandas.

Groups customers by the period of their first recorded activity, then tracks
what fraction of each cohort is still active in each subsequent period —
the standard retention-cohort table, computed directly from the real data.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class CohortResult:
    cohort_sizes: pd.Series
    retention_counts: pd.DataFrame  # index=cohort period, columns=periods since cohort start, values=active customer count
    retention_pct: pd.DataFrame


def build_retention_cohorts(df: pd.DataFrame, customer_col: str, date_col: str, period: str = "M") -> CohortResult:
    working = df[[customer_col, date_col]].dropna().copy()
    working[date_col] = pd.to_datetime(working[date_col])
    working["activity_period"] = working[date_col].dt.to_period(period)

    if working.empty:
        raise ValueError("No valid rows with both a customer ID and a date")

    first_activity = working.groupby(customer_col)["activity_period"].min().rename("cohort_period")
    working = working.merge(first_activity, on=customer_col)
    working["period_number"] = (working["activity_period"] - working["cohort_period"]).apply(lambda x: x.n)

    cohort_data = working.groupby(["cohort_period", "period_number"])[customer_col].nunique().reset_index()
    retention_counts = cohort_data.pivot(index="cohort_period", columns="period_number", values=customer_col)

    cohort_sizes = retention_counts[0].copy()
    retention_pct = retention_counts.divide(cohort_sizes, axis=0) * 100

    retention_counts.index = retention_counts.index.astype(str)
    retention_pct.index = retention_pct.index.astype(str)
    cohort_sizes.index = cohort_sizes.index.astype(str)

    return CohortResult(cohort_sizes=cohort_sizes, retention_counts=retention_counts, retention_pct=retention_pct.round(1))
