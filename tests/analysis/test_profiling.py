import pandas as pd

from analysis import profiling


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
