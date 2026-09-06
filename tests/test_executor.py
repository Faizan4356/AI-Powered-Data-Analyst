import pandas as pd

from core.executor import execute_plan
from core.validation import OperationPlan


def make_df():
    return pd.DataFrame(
        {
            "region": ["A", "B", "A", "B"],
            "revenue": [100, 200, 300, 400],
        }
    )


def test_groupby_sum():
    df = make_df()
    plan = OperationPlan(op_type="groupby", groupby=["region"], agg_func="sum", agg_column="revenue")
    result = execute_plan(plan, df)
    assert result.error is None
    row_a = result.result_df[result.result_df["region"] == "A"]
    assert row_a["revenue"].iloc[0] == 400


def test_scalar_aggregation():
    df = make_df()
    plan = OperationPlan(op_type="aggregation", agg_func="mean", agg_column="revenue")
    result = execute_plan(plan, df)
    assert result.scalar_value == 250.0


def test_filters_applied():
    df = make_df()
    plan = OperationPlan(op_type="filter", filters=[{"column": "region", "op": "==", "value": "A"}], columns=["region", "revenue"])
    result = execute_plan(plan, df)
    assert len(result.result_df) == 2
    assert set(result.result_df["region"]) == {"A"}


def test_raw_sql_execution():
    df = make_df()
    plan = OperationPlan(op_type="raw_sql", raw_sql="SELECT region, SUM(revenue) as total FROM df GROUP BY region")
    result = execute_plan(plan, df)
    assert result.error is None
    assert set(result.result_df["region"]) == {"A", "B"}


def test_raw_sql_blocks_injection():
    df = make_df()
    plan = OperationPlan(op_type="raw_sql", raw_sql="SELECT * FROM df; DROP TABLE df")
    result = execute_plan(plan, df)
    assert result.error is not None


def test_describe_op():
    df = make_df()
    plan = OperationPlan(op_type="describe", columns=["revenue"])
    result = execute_plan(plan, df)
    assert result.error is None
    assert "index" in result.result_df.columns


def test_filter_operators_gt_in_contains():
    df = make_df()
    gt_plan = OperationPlan(op_type="filter", filters=[{"column": "revenue", "op": ">", "value": 200}], columns=["region", "revenue"])
    assert len(execute_plan(gt_plan, df).result_df) == 2

    in_plan = OperationPlan(op_type="filter", filters=[{"column": "region", "op": "in", "value": ["A"]}], columns=["region"])
    assert len(execute_plan(in_plan, df).result_df) == 2

    not_in_plan = OperationPlan(op_type="filter", filters=[{"column": "region", "op": "not in", "value": ["A"]}], columns=["region"])
    assert len(execute_plan(not_in_plan, df).result_df) == 2

    contains_plan = OperationPlan(op_type="filter", filters=[{"column": "region", "op": "contains", "value": "a"}], columns=["region"])
    assert len(execute_plan(contains_plan, df).result_df) == 2


def test_sort_and_limit_on_non_grouped_result():
    df = make_df()
    plan = OperationPlan(op_type="filter", columns=["region", "revenue"], sort_by="revenue", sort_desc=True, limit=2)
    result = execute_plan(plan, df)
    assert len(result.result_df) == 2
    assert result.result_df["revenue"].iloc[0] == 400


def test_scalar_aggregation_with_limit_is_noop():
    df = make_df()
    plan = OperationPlan(op_type="aggregation", agg_func="sum", agg_column="revenue", limit=5)
    result = execute_plan(plan, df)
    assert result.scalar_value == 1000


def test_execute_plan_catches_errors():
    df = make_df()
    plan = OperationPlan(op_type="describe", columns=["does_not_exist"])
    result = execute_plan(plan, df)
    assert result.error is not None
