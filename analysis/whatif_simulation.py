"""What-if / scenario simulation: adjust one input variable and see the
projected impact using a real fitted linear regression model.

Always clearly labeled as a simulation/projection, never presented as a
measured fact — the number comes from a fitted model's prediction, not from
the LLM, and the caller must not blur that distinction downstream.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class WhatIfResult:
    baseline_input: float
    scenario_input: float
    baseline_prediction: float
    scenario_prediction: float
    pct_change: float
    r_squared: float
    summary: str


def simulate_variable_change(df: pd.DataFrame, input_col: str, target_col: str, pct_change: float) -> WhatIfResult:
    working = df[[input_col, target_col]].dropna()
    if len(working) < 5:
        raise ValueError("Need at least 5 rows with both columns present to fit a simulation model")

    X = working[[input_col]].values
    y = working[target_col].values

    model = LinearRegression().fit(X, y)
    r_squared = float(model.score(X, y))

    baseline_input = float(working[input_col].mean())
    scenario_input = baseline_input * (1 + pct_change / 100)

    baseline_prediction = float(model.predict([[baseline_input]])[0])
    scenario_prediction = float(model.predict([[scenario_input]])[0])
    result_pct_change = (
        (scenario_prediction - baseline_prediction) / baseline_prediction * 100 if baseline_prediction else float("inf")
    )

    fit_caveat = " (weak model fit — treat this projection with caution)" if r_squared < 0.3 else ""
    summary = (
        f"SIMULATION, not measured data: if {input_col} changes by {pct_change:+.1f}% "
        f"(from {baseline_input:.2f} to {scenario_input:.2f}), the fitted linear model projects "
        f"{target_col} to change by {result_pct_change:+.1f}% (from {baseline_prediction:.2f} to "
        f"{scenario_prediction:.2f}). Model fit: R²={r_squared:.3f}{fit_caveat}."
    )

    return WhatIfResult(
        baseline_input=round(baseline_input, 4),
        scenario_input=round(scenario_input, 4),
        baseline_prediction=round(baseline_prediction, 4),
        scenario_prediction=round(scenario_prediction, 4),
        pct_change=round(result_pct_change, 2),
        r_squared=round(r_squared, 4),
        summary=summary,
    )
