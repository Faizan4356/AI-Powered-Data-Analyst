import pandas as pd

from analysis import anomaly_detection, profiling, segmentation


def test_profiling_flags_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 2, None, 2], "b": ["x", "y", "z", "y"]})
    report = profiling.profile_dataframe(df)
    assert report.duplicate_rows == 1
    a_profile = next(c for c in report.columns if c.name == "a")
    assert a_profile.missing_count == 1


def test_profiling_detects_pii():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com", "c@z.com"]})
    report = profiling.profile_dataframe(df)
    assert report.columns[0].likely_pii == "email"


def test_quality_score_perfect_for_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    report = profiling.profile_dataframe(df)
    assert report.quality_score == 100.0
    assert report.completeness_score == 100.0
    assert report.consistency_score == 100.0
    assert report.validity_score == 100.0


def test_quality_score_penalizes_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, None, None, 1], "b": ["x", "y", "y", "x"]})
    report = profiling.profile_dataframe(df)
    assert report.completeness_score < 100.0
    assert report.consistency_score < 100.0
    assert report.quality_score < 100.0


def test_apply_cleaning_drops_duplicates_and_columns():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    cleaned = profiling.apply_cleaning(df, {"drop_duplicates": True, "drop_columns": ["b"]})
    assert len(cleaned) == 2
    assert "b" not in cleaned.columns


def test_zscore_anomaly_detection():
    df = pd.DataFrame({"v": [10, 11, 9, 10, 10, 9, 11, 10, 5000]})
    result = anomaly_detection.detect_zscore(df, "v", threshold=2.0)
    assert result.total_flagged >= 1


def test_rfm_and_kmeans_segmentation():
    df = pd.DataFrame(
        {
            "customer": ["a", "a", "b", "c", "c", "c"],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15", "2024-01-01", "2024-01-10", "2024-01-20"]),
            "amount": [10, 20, 5, 100, 100, 100],
        }
    )
    rfm = segmentation.compute_rfm(df, "customer", "date", "amount")
    assert set(rfm["customer"]) == {"a", "b", "c"}
    seg = segmentation.kmeans_segment(rfm, n_clusters=2)
    assert seg.n_clusters == 2
    assert "label" in seg.labeled_table.columns


def test_compute_rfm_rejects_duplicate_column_selection():
    """If customer/date/amount aren't three distinct columns (e.g. two UI
    dropdowns left on the same default), pandas would otherwise fail deep
    inside pd.to_datetime with a cryptic 'duplicate keys' error."""
    import pytest

    df = pd.DataFrame({"customer": ["a", "b"], "amount": [10, 20]})
    with pytest.raises(ValueError, match="distinct columns"):
        segmentation.compute_rfm(df, "customer", "customer", "amount")
