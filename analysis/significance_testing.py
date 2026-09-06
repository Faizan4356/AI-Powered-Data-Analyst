"""A/B testing / statistical significance testing. Pure scipy — no LLM.

Compares two segments or two time periods on a metric and reports whether
the difference is statistically significant, in plain language derived
directly from the computed p-value (never phrased by an LLM as a guess).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


@dataclass
class TTestResult:
    group_a_label: str
    group_b_label: str
    group_a_n: int
    group_b_n: int
    group_a_mean: float
    group_b_mean: float
    t_statistic: float
    p_value: float
    significant: bool
    cohens_d: float
    summary: str


@dataclass
class ChiSquareResult:
    contingency_table: pd.DataFrame
    chi2_statistic: float
    p_value: float
    degrees_of_freedom: int
    significant: bool
    summary: str


def compare_means(
    df: pd.DataFrame, group_col: str, value_col: str, group_a: str, group_b: str, alpha: float = ALPHA
) -> TTestResult:
    a_values = df.loc[df[group_col] == group_a, value_col].dropna()
    b_values = df.loc[df[group_col] == group_b, value_col].dropna()

    if len(a_values) < 2 or len(b_values) < 2:
        raise ValueError("Each group needs at least 2 observations for a t-test")

    t_stat, p_value = stats.ttest_ind(a_values, b_values, equal_var=False)

    pooled_std = np.sqrt(((len(a_values) - 1) * a_values.std(ddof=1) ** 2 + (len(b_values) - 1) * b_values.std(ddof=1) ** 2) / (len(a_values) + len(b_values) - 2))
    cohens_d = (a_values.mean() - b_values.mean()) / pooled_std if pooled_std else 0.0

    significant = bool(p_value < alpha)
    summary = _ttest_summary(group_a, group_b, a_values.mean(), b_values.mean(), p_value, significant, alpha)

    return TTestResult(
        group_a_label=str(group_a),
        group_b_label=str(group_b),
        group_a_n=len(a_values),
        group_b_n=len(b_values),
        group_a_mean=round(float(a_values.mean()), 4),
        group_b_mean=round(float(b_values.mean()), 4),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_value), 6),
        significant=significant,
        cohens_d=round(float(cohens_d), 4),
        summary=summary,
    )


def compare_time_periods(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period_a_start,
    period_a_end,
    period_b_start,
    period_b_end,
    alpha: float = ALPHA,
) -> TTestResult:
    dates = pd.to_datetime(df[date_col])
    a_mask = (dates >= pd.Timestamp(period_a_start)) & (dates <= pd.Timestamp(period_a_end))
    b_mask = (dates >= pd.Timestamp(period_b_start)) & (dates <= pd.Timestamp(period_b_end))

    working = df.copy()
    working["_period_label"] = np.where(a_mask, "period_a", np.where(b_mask, "period_b", None))
    working = working.dropna(subset=["_period_label"])

    return compare_means(working, "_period_label", value_col, "period_a", "period_b", alpha=alpha)


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str, alpha: float = ALPHA) -> ChiSquareResult:
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    significant = bool(p_value < alpha)
    summary = _chi2_summary(col_a, col_b, p_value, significant, alpha)

    return ChiSquareResult(
        contingency_table=contingency,
        chi2_statistic=round(float(chi2), 4),
        p_value=round(float(p_value), 6),
        degrees_of_freedom=int(dof),
        significant=significant,
        summary=summary,
    )


def _ttest_summary(label_a, label_b, mean_a, mean_b, p_value, significant, alpha) -> str:
    direction = "higher" if mean_a > mean_b else "lower"
    verdict = (
        f"statistically significant (p={p_value:.4f} < {alpha})"
        if significant
        else f"not statistically significant (p={p_value:.4f} >= {alpha})"
    )
    return (
        f"'{label_a}' (mean={mean_a:.4f}) is {direction} than '{label_b}' (mean={mean_b:.4f}). "
        f"This difference is {verdict}."
    )


def _chi2_summary(col_a, col_b, p_value, significant, alpha) -> str:
    verdict = (
        f"statistically significant (p={p_value:.4f} < {alpha})"
        if significant
        else f"not statistically significant (p={p_value:.4f} >= {alpha})"
    )
    relation = "an association" if significant else "no association"
    return f"There is {relation} between '{col_a}' and '{col_b}'. This result is {verdict}."
