import numpy as np
import pandas as pd
import pytest

from analysis.whatif_simulation import simulate_variable_change


def test_simulate_variable_change_with_perfect_linear_relationship():
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50], "revenue": [100, 200, 300, 400, 500]})
    result = simulate_variable_change(df, "price", "revenue", pct_change=10)
    assert result.r_squared > 0.99
    assert result.pct_change == pytest.approx(10.0, abs=0.5)
    assert "SIMULATION" in result.summary


def test_simulate_variable_change_negative_pct():
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50], "revenue": [100, 200, 300, 400, 500]})
    result = simulate_variable_change(df, "price", "revenue", pct_change=-20)
    assert result.scenario_prediction < result.baseline_prediction


def test_simulate_variable_change_flags_weak_fit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    result = simulate_variable_change(df, "x", "y", pct_change=10)
    assert result.r_squared < 0.3
    assert "weak model fit" in result.summary


def test_simulate_variable_change_raises_on_too_few_rows():
    df = pd.DataFrame({"price": [10, 20], "revenue": [100, 200]})
    with pytest.raises(ValueError):
        simulate_variable_change(df, "price", "revenue", pct_change=10)
