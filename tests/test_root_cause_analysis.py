import pandas as pd
import pytest

from analysis.root_cause_analysis import decompose_change


def make_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-05", "2024-01-10", "2024-01-15", "2024-02-05", "2024-02-10", "2024-02-15"]
            ),
            "region": ["East", "West", "East", "East", "West", "East"],
            "revenue": [100, 50, 100, 40, 55, 60],  # Jan total=250, Feb total=155
        }
    )


def test_decompose_change_computes_correct_totals():
    result = decompose_change(make_df(), "date", "revenue", "region", "2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29")
    assert result.total_a == 250.0
    assert result.total_b == 155.0
    assert result.total_change == -95.0
    assert round(result.total_pct_change, 1) == -38.0


def test_decompose_change_identifies_top_driver():
    result = decompose_change(make_df(), "date", "revenue", "region", "2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29")
    # East: Jan=200 (100+100), Feb=100 (40+60) -> change=-100
    # West: Jan=50, Feb=55 -> change=+5
    top = result.drivers[0]
    assert top.dimension_value == "East"
    assert top.change == -100.0
    assert "East" in result.summary


def test_decompose_change_pct_of_total_sums_reasonably():
    result = decompose_change(make_df(), "date", "revenue", "region", "2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29")
    total_pct = sum(d.pct_of_total_change for d in result.drivers)
    assert round(total_pct, 1) == 100.0


def test_decompose_change_raises_on_empty_period():
    with pytest.raises(ValueError):
        decompose_change(make_df(), "date", "revenue", "region", "2023-01-01", "2023-01-31", "2024-02-01", "2024-02-29")
