"""Statistical (z-score/IQR) and model-based (Isolation Forest) anomaly detection."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    flagged_rows: pd.DataFrame
    method: str
    total_flagged: int
    flagged_pct: float


def detect_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> AnomalyResult:
    series = df[column]
    z = (series - series.mean()) / series.std(ddof=0)
    flagged_mask = z.abs() > threshold
    flagged = df[flagged_mask].copy()
    flagged["zscore"] = z[flagged_mask]
    return AnomalyResult(
        flagged_rows=flagged,
        method="z-score",
        total_flagged=int(flagged_mask.sum()),
        flagged_pct=round(flagged_mask.mean() * 100, 2) if len(df) else 0.0,
    )


def detect_iqr(df: pd.DataFrame, column: str) -> AnomalyResult:
    series = df[column]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged_mask = (series < lower) | (series > upper)
    flagged = df[flagged_mask].copy()
    return AnomalyResult(
        flagged_rows=flagged,
        method="IQR",
        total_flagged=int(flagged_mask.sum()),
        flagged_pct=round(flagged_mask.mean() * 100, 2) if len(df) else 0.0,
    )


def detect_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.05) -> AnomalyResult:
    working = df[columns].dropna()
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(working)
    flagged_mask = preds == -1
    flagged = working[flagged_mask].copy()
    flagged["anomaly_score"] = model.decision_function(working)[flagged_mask]
    return AnomalyResult(
        flagged_rows=flagged,
        method="Isolation Forest",
        total_flagged=int(flagged_mask.sum()),
        flagged_pct=round(flagged_mask.mean() * 100, 2) if len(working) else 0.0,
    )
