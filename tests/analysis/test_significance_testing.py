import numpy as np
import pandas as pd
import pytest

from analysis.significance_testing import chi_square_test, compare_means, compare_time_periods


def test_compare_means_detects_significant_difference():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "value": list(rng.normal(50, 5, 100)) + list(rng.normal(70, 5, 100)),
        }
    )
    result = compare_means(df, "group", "value", "A", "B")
    assert result.significant is True
    assert result.p_value < 0.05
    assert result.group_a_mean < result.group_b_mean
    assert "statistically significant" in result.summary


def test_compare_means_no_significant_difference_for_identical_distributions():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "group": ["A"] * 200 + ["B"] * 200,
            "value": list(rng.normal(50, 10, 200)) + list(rng.normal(50, 10, 200)),
        }
    )
    result = compare_means(df, "group", "value", "A", "B")
    assert result.significant is False


def test_compare_means_raises_on_too_few_observations():
    df = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})
    with pytest.raises(ValueError):
        compare_means(df, "group", "value", "A", "B")


def test_compare_time_periods_slices_by_date():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"date": dates, "value": list(range(10)) + list(range(100, 110))})
    result = compare_time_periods(df, "date", "value", "2024-01-01", "2024-01-10", "2024-01-11", "2024-01-20")
    assert result.group_a_n == 10
    assert result.group_b_n == 10
    assert result.significant is True


def test_chi_square_detects_association():
    df = pd.DataFrame(
        {
            "channel": ["email"] * 80 + ["ads"] * 80,
            "converted": ["yes"] * 60 + ["no"] * 20 + ["yes"] * 10 + ["no"] * 70,
        }
    )
    result = chi_square_test(df, "channel", "converted")
    assert result.significant is True
    assert "association" in result.summary


def test_chi_square_no_association_for_uniform_distribution():
    df = pd.DataFrame(
        {
            "channel": (["email"] * 50 + ["ads"] * 50) * 2,
            "converted": (["yes"] * 25 + ["no"] * 25) * 4,
        }
    )
    result = chi_square_test(df, "channel", "converted")
    assert result.significant is False
