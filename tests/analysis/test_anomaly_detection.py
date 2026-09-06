import pandas as pd

from analysis import anomaly_detection


def test_zscore_anomaly_detection():
    df = pd.DataFrame({"v": [10, 11, 9, 10, 10, 9, 11, 10, 5000]})
    result = anomaly_detection.detect_zscore(df, "v", threshold=2.0)
    assert result.total_flagged >= 1
