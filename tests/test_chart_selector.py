import pandas as pd

from core.chart_selector import choose_chart


def test_empty_result_returns_table():
    spec = choose_chart(pd.DataFrame())
    assert spec.chart_type == "table"


def test_single_row_returns_table():
    df = pd.DataFrame({"revenue": [500]})
    spec = choose_chart(df)
    assert spec.chart_type == "table"


def test_date_and_numeric_returns_line():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=5), "revenue": [10, 20, 15, 30, 25]})
    spec = choose_chart(df)
    assert spec.chart_type == "line"
    assert spec.x == "ds" and spec.y == "revenue"


def test_low_cardinality_category_and_numeric_returns_bar():
    df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [10, 20, 30]})
    spec = choose_chart(df)
    assert spec.chart_type == "bar"
    assert spec.x == "region" and spec.y == "revenue"


def test_high_cardinality_category_returns_table():
    df = pd.DataFrame({"customer_id": [f"c{i}" for i in range(50)], "revenue": range(50)})
    spec = choose_chart(df)
    assert spec.chart_type == "table"


def test_two_category_dims_and_numeric_returns_heatmap():
    df = pd.DataFrame({"region": ["A", "A", "B", "B"], "product": ["X", "Y", "X", "Y"], "revenue": [1, 2, 3, 4]})
    spec = choose_chart(df)
    assert spec.chart_type == "heatmap"


def test_two_numeric_cols_returns_scatter():
    df = pd.DataFrame({"price": [1, 2, 3, 4], "quantity": [10, 20, 30, 40]})
    spec = choose_chart(df)
    assert spec.chart_type == "scatter"


def test_single_numeric_many_rows_returns_histogram():
    df = pd.DataFrame({"revenue": list(range(20))})
    spec = choose_chart(df)
    assert spec.chart_type == "histogram"
