"""Explainable time-series forecasting using statsmodels (Holt-Winters).

Purely statistical — no LLM involvement in the numbers. The LLM is only used
downstream (response_composer) to narrate the already-computed forecast.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    history: pd.DataFrame  # columns: ds, y
    forecast: pd.DataFrame  # columns: ds, yhat, yhat_lower, yhat_upper
    trend_direction: str
    seasonality_detected: bool
    confidence_note: str


def run_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 12, freq: str = "ME") -> ForecastResult:
    series_df = df[[date_col, value_col]].dropna().copy()
    series_df[date_col] = pd.to_datetime(series_df[date_col])
    series_df = series_df.sort_values(date_col)
    series_df = series_df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum().reset_index()
    series_df = series_df.rename(columns={date_col: "ds", value_col: "y"})

    if len(series_df) < 6:
        raise ValueError("Need at least 6 historical periods to forecast reliably")

    seasonal_periods = 12 if freq.upper().startswith("M") else 7
    use_seasonal = len(series_df) >= 2 * seasonal_periods

    model = ExponentialSmoothing(
        series_df["y"],
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    )
    fit = model.fit()

    forecast_values = fit.forecast(periods)
    resid_std = np.std(fit.resid)
    last_date = series_df["ds"].iloc[-1]
    future_dates = pd.date_range(last_date, periods=periods + 1, freq=freq)[1:]

    forecast_df = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": forecast_values.values,
            "yhat_lower": forecast_values.values - 1.96 * resid_std,
            "yhat_upper": forecast_values.values + 1.96 * resid_std,
        }
    )

    trend_direction = "increasing" if forecast_values.iloc[-1] > series_df["y"].iloc[-1] else "decreasing"

    return ForecastResult(
        history=series_df,
        forecast=forecast_df,
        trend_direction=trend_direction,
        seasonality_detected=use_seasonal,
        confidence_note="95% confidence interval based on in-sample residual standard deviation",
    )
