import pandas as pd
import pytest

from analysis.cohort_analysis import build_retention_cohorts


def make_df():
    return pd.DataFrame(
        {
            "customer": ["a", "a", "a", "b", "b", "c", "c", "c", "c"],
            "date": pd.to_datetime(
                [
                    "2024-01-05",  # a's first (Jan cohort)
                    "2024-02-10",  # a active in Feb
                    "2024-03-15",  # a active in Mar
                    "2024-02-01",  # b's first (Feb cohort)
                    "2024-02-20",  # b active again in Feb (same period, dedup by nunique per period)
                    "2024-01-02",  # c's first (Jan cohort)
                    "2024-02-05",  # c active in Feb
                    "2024-03-01",  # c active in Mar
                    "2024-04-01",  # c active in Apr
                ]
            ),
        }
    )


def test_cohort_sizes_reflect_first_activity_period():
    result = build_retention_cohorts(make_df(), "customer", "date", period="M")
    # January cohort = customers a and c
    assert result.cohort_sizes["2024-01"] == 2
    # February cohort = customer b only
    assert result.cohort_sizes["2024-02"] == 1


def test_retention_counts_track_repeat_activity():
    result = build_retention_cohorts(make_df(), "customer", "date", period="M")
    jan_row = result.retention_counts.loc["2024-01"]
    assert jan_row[0] == 2  # a, c active in their own first month
    assert jan_row[1] == 2  # both a and c active one month later (Feb)
    assert jan_row[2] == 2  # both a (Mar) and c (Mar) active two months later
    assert jan_row[3] == 1  # only c active three months later (Apr)


def test_retention_pct_is_normalized_by_cohort_size():
    result = build_retention_cohorts(make_df(), "customer", "date", period="M")
    jan_row_pct = result.retention_pct.loc["2024-01"]
    assert jan_row_pct[0] == 100.0
    assert jan_row_pct[1] == 100.0
    assert jan_row_pct[2] == 100.0
    assert jan_row_pct[3] == 50.0


def test_raises_on_empty_input():
    df = pd.DataFrame({"customer": [None], "date": [None]})
    with pytest.raises(ValueError):
        build_retention_cohorts(df, "customer", "date")
