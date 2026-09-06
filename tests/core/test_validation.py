import pandas as pd
import pytest

from core.validation import ValidationError, validate_plan, validate_sql, sanitize_column_name


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["A", "B", "A"], "revenue": [10, 20, 30]})


def test_validate_plan_unknown_column_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "aggregation", "columns": ["nope"]}, df)


def test_validate_plan_unknown_op_type_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "hack"}, df)


def test_validate_plan_valid_groupby(df):
    plan = validate_plan(
        {"op_type": "groupby", "groupby": ["region"], "agg_func": "sum", "agg_column": "revenue"}, df
    )
    assert plan.groupby == ["region"]
    assert plan.agg_func == "sum"


def test_validate_plan_bad_agg_func_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "aggregation", "agg_func": "os.system", "agg_column": "revenue"}, df)


def test_validate_sql_blocks_drop():
    with pytest.raises(ValidationError):
        validate_sql("DROP TABLE df")


def test_validate_sql_blocks_multistatement():
    with pytest.raises(ValidationError):
        validate_sql("SELECT * FROM df; DROP TABLE df")


def test_validate_sql_allows_select():
    assert validate_sql("SELECT * FROM df WHERE revenue > 10") == "SELECT * FROM df WHERE revenue > 10"


def test_sanitize_column_name():
    assert sanitize_column_name("Revenue ($)") == "Revenue"
    assert sanitize_column_name("2024 sales") == "c_2024_sales"
    assert sanitize_column_name("") == "col"


def test_validate_plan_bad_filter_column_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "filter", "filters": [{"column": "nope", "op": "==", "value": 1}]}, df)


def test_validate_plan_bad_filter_op_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "filter", "filters": [{"column": "revenue", "op": "; DROP", "value": 1}]}, df)


def test_validate_plan_bad_agg_column_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "aggregation", "agg_func": "sum", "agg_column": "nope"}, df)


def test_validate_plan_bad_sort_column_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan({"op_type": "aggregation", "sort_by": "nope"}, df)


def test_validate_plan_limit_clamped(df):
    plan = validate_plan({"op_type": "describe", "limit": 999999}, df)
    assert plan.limit == 10000


def test_validate_plan_embeds_raw_sql(df):
    plan = validate_plan({"op_type": "raw_sql", "raw_sql": "SELECT * FROM df"}, df)
    assert plan.raw_sql == "SELECT * FROM df"


def test_validate_sql_blocks_non_select():
    with pytest.raises(ValidationError):
        validate_sql("PRAGMA table_info(df)")
