"""Multi-table join planning: key inference and merge execution are pure
Pandas (no LLM); only the choice of which join to run goes through a mocked
LLM here, and only from a pre-verified candidate list.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.join_planner import compute_pairwise_overlap_matrix, execute_joins, infer_join_candidates, plan_joins
from core.llm_client import LLMResponse
from core.validation import JoinStep, ValidationError, validate_join_plan


@pytest.fixture
def tables():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Ann", "Bo", "Cy"]})
    orders = pd.DataFrame({"order_id": [10, 11, 12], "customer_id": [1, 2, 1], "amount": [50, 75, 20]})
    return {"customers": customers, "orders": orders}


def make_llm(text):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text=text, latency_ms=1.0, tokens_estimate=1, backend="mock")
    return llm


def test_compute_pairwise_overlap_matrix_shows_full_shape(tables):
    matrix = compute_pairwise_overlap_matrix(tables["customers"], tables["orders"])
    assert list(matrix.index) == ["customer_id", "name"]
    assert list(matrix.columns) == ["order_id", "customer_id", "amount"]
    # customers.customer_id values {1,2,3} vs orders.customer_id values {1,2} -> overlap 2/2 = 1.0
    assert matrix.loc["customer_id", "customer_id"] == 1.0
    # unrelated columns still get a real (low) score, not omitted
    assert matrix.loc["name", "order_id"] == 0.0


def test_compute_pairwise_overlap_matrix_handles_empty_columns():
    left = pd.DataFrame({"a": [None, None]})
    right = pd.DataFrame({"b": [1, 2]})
    matrix = compute_pairwise_overlap_matrix(left, right)
    assert matrix.loc["a", "b"] == 0.0


def test_infer_join_candidates_finds_real_overlap(tables):
    candidates = infer_join_candidates(tables)
    assert any(c.left_column == "customer_id" and c.right_column == "customer_id" for c in candidates)


def test_infer_join_candidates_ignores_unrelated_columns():
    tables = {
        "a": pd.DataFrame({"x": [1, 2, 3]}),
        "b": pd.DataFrame({"y": [100, 200, 300]}),
    }
    candidates = infer_join_candidates(tables)
    assert candidates == []


def test_validate_join_plan_rejects_unknown_table(tables):
    with pytest.raises(ValidationError):
        validate_join_plan({"joins": [{"left_table": "ghost", "right_table": "orders", "left_key": "x", "right_key": "y"}]}, tables)


def test_validate_join_plan_rejects_unknown_column(tables):
    with pytest.raises(ValidationError):
        validate_join_plan(
            {"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "made_up", "right_key": "customer_id"}]},
            tables,
        )


def test_validate_join_plan_rejects_bad_how(tables):
    with pytest.raises(ValidationError):
        validate_join_plan(
            {"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "how": "cross"}]},
            tables,
        )


def test_validate_join_plan_rejects_non_candidate_column_pair(tables):
    """Both columns exist individually, but this pair was never verified to
    actually overlap — the LLM must not be able to join on it anyway."""
    candidates = infer_join_candidates(tables)
    with pytest.raises(ValidationError):
        validate_join_plan(
            {"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "name", "right_key": "amount", "how": "inner"}]},
            tables,
            candidates=candidates,
        )


def test_validate_join_plan_accepts_valid_plan(tables):
    steps = validate_join_plan(
        {"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "how": "inner"}]},
        tables,
    )
    assert len(steps) == 1
    assert steps[0].how == "inner"


def test_plan_joins_uses_llm_choice(tables):
    candidates = infer_join_candidates(tables)
    llm = make_llm('{"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "customer_id", "right_key": "customer_id", "how": "left"}]}')
    steps = plan_joins(tables, candidates, llm)
    assert steps[0].how == "left"


def test_plan_joins_falls_back_when_llm_unavailable(tables):
    from core.llm_client import LLMUnavailableError

    candidates = infer_join_candidates(tables)
    llm = MagicMock()
    llm.complete.side_effect = LLMUnavailableError("down")
    steps = plan_joins(tables, candidates, llm)
    assert steps[0].left_key == "customer_id"
    assert steps[0].how == "inner"


def test_plan_joins_falls_back_when_llm_invents_bad_plan(tables):
    """LLM proposes a join on a column that isn't a real candidate -> rejected -> fallback to top candidate."""
    candidates = infer_join_candidates(tables)
    llm = make_llm('{"joins": [{"left_table": "customers", "right_table": "orders", "left_key": "name", "right_key": "amount", "how": "inner"}]}')
    steps = plan_joins(tables, candidates, llm)
    assert steps[0].left_key == "customer_id"


def test_plan_joins_raises_when_no_candidates():
    tables = {"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"y": [2]})}
    llm = make_llm("{}")
    with pytest.raises(ValidationError):
        plan_joins(tables, [], llm)


def test_execute_joins_merges_correctly(tables):
    steps = [JoinStep(left_table="customers", right_table="orders", left_key="customer_id", right_key="customer_id", how="inner")]
    result = execute_joins(tables, steps)
    assert result.error is None
    assert len(result.merged_df) == 3
    assert "amount" in result.merged_df.columns and "name" in result.merged_df.columns


def test_execute_joins_handles_error_gracefully(tables):
    bad_step = JoinStep(left_table="customers", right_table="does_not_exist", left_key="customer_id", right_key="customer_id", how="inner")
    result = execute_joins(tables, [bad_step])
    assert result.error is not None
