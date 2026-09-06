import pandas as pd
import pytest

from analysis import segmentation


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
    df = pd.DataFrame({"customer": ["a", "b"], "amount": [10, 20]})
    with pytest.raises(ValueError, match="distinct columns"):
        segmentation.compute_rfm(df, "customer", "customer", "amount")
