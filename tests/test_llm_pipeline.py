"""Tests for the LLM-facing stages using a mocked LLMClient — no network calls.

These exist to prove the two guarantees the whole app is built on:
1. If the LLM is unreachable, every stage degrades to a deterministic fallback
   instead of crashing.
2. If the LLM returns garbage/invalid/out-of-schema JSON, validation.py catches
   it before anything executes against the real data.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.executor import ExecutionResult
from core.intent_parser import parse_intent
from core.llm_client import LLMResponse, LLMUnavailableError
from core.operation_planner import plan_operation
from core.response_composer import compose_response


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["A", "B", "A"], "revenue": [100, 200, 300]})


def make_llm(text=None, raises=False):
    llm = MagicMock()
    if raises:
        llm.complete.side_effect = LLMUnavailableError("mock LLM down")
    else:
        llm.complete.return_value = LLMResponse(text=text, latency_ms=1.0, tokens_estimate=1, backend="mock")
    return llm


# ---------------- intent_parser ----------------

def test_parse_intent_uses_llm_output():
    llm = make_llm(text='{"intent": "trend", "reasoning": "mentions over time"}')
    intent = parse_intent("How did revenue trend over time?", llm)
    assert intent.intent_type == "trend"


def test_parse_intent_falls_back_when_llm_unavailable():
    llm = make_llm(raises=True)
    intent = parse_intent("anything", llm)
    assert intent.intent_type == "aggregation"
    assert "LLM unavailable" in intent.reasoning


def test_parse_intent_rejects_unknown_intent_type():
    llm = make_llm(text='{"intent": "hack_the_planet", "reasoning": "n/a"}')
    intent = parse_intent("anything", llm)
    assert intent.intent_type == "aggregation"


# ---------------- operation_planner ----------------

def test_plan_operation_validates_llm_plan(df):
    from core.intent_parser import Intent

    llm = make_llm(text='{"op_type": "groupby", "groupby": ["region"], "agg_func": "sum", "agg_column": "revenue"}')
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="sum revenue by region")
    plan = plan_operation(intent, df, llm)
    assert plan.op_type == "groupby"
    assert plan.groupby == ["region"]


def test_plan_operation_falls_back_on_invalid_column(df):
    """LLM hallucinates a column that doesn't exist -> validation rejects it -> safe fallback plan."""
    from core.intent_parser import Intent

    llm = make_llm(text='{"op_type": "aggregation", "agg_func": "sum", "agg_column": "made_up_column"}')
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="sum made_up_column")
    plan = plan_operation(intent, df, llm)
    assert plan.op_type == "describe"  # safe fallback, never executes against a fake column


def test_plan_operation_falls_back_on_malformed_json(df):
    from core.intent_parser import Intent

    llm = make_llm(text="not json at all")
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="anything")
    plan = plan_operation(intent, df, llm)
    assert plan.op_type == "describe"


def test_plan_operation_falls_back_when_llm_unavailable(df):
    from core.intent_parser import Intent

    llm = make_llm(raises=True)
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="anything")
    plan = plan_operation(intent, df, llm)
    assert plan.op_type == "describe"


def test_plan_operation_includes_error_context_in_retry_prompt(df):
    from core.intent_parser import Intent

    llm = make_llm(text='{"op_type": "describe"}')
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="total revenue")
    plan_operation(intent, df, llm, error_context="Unknown column: made_up")

    sent_user_prompt = llm.complete.call_args[0][1]
    assert "Unknown column: made_up" in sent_user_prompt


def test_plan_operation_rejects_sql_injection_attempt(df):
    """LLM tries to smuggle destructive SQL into raw_sql -> validation blocks it -> fallback plan."""
    from core.intent_parser import Intent

    llm = make_llm(text='{"op_type": "aggregation", "raw_sql": "SELECT * FROM df; DROP TABLE df"}')
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="anything")
    plan = plan_operation(intent, df, llm)
    assert plan.op_type == "describe"


# ---------------- response_composer ----------------

def test_compose_response_uses_llm_text():
    llm = make_llm(text="Total revenue is 600, driven mostly by region A.")
    result = ExecutionResult(op_type="aggregation", result_df=pd.DataFrame({"revenue": [600]}), code_executed="df['revenue'].sum()", scalar_value=600)
    answer = compose_response("what is total revenue", result, llm)
    assert "600" in answer


def test_compose_response_falls_back_on_llm_outage():
    llm = make_llm(raises=True)
    result = ExecutionResult(op_type="aggregation", result_df=pd.DataFrame({"revenue": [600]}), code_executed="df['revenue'].sum()", scalar_value=600)
    answer = compose_response("what is total revenue", result, llm)
    assert "600" in answer  # templated fallback still surfaces the real computed number


def test_compose_response_surfaces_execution_error_without_calling_llm():
    llm = make_llm(text="should not be used")
    result = ExecutionResult(op_type="aggregation", result_df=pd.DataFrame(), code_executed="", error="Unknown column: foo")
    answer = compose_response("anything", result, llm)
    assert "Unknown column: foo" in answer
    llm.complete.assert_not_called()


def test_compose_response_empty_table_fallback_on_llm_outage():
    llm = make_llm(raises=True)
    result = ExecutionResult(op_type="groupby", result_df=pd.DataFrame(), code_executed="df.groupby(...)")
    answer = compose_response("anything", result, llm)
    assert "No rows matched" in answer


def test_compose_response_table_fallback_on_llm_outage():
    llm = make_llm(raises=True)
    result = ExecutionResult(op_type="groupby", result_df=pd.DataFrame({"region": ["A"], "revenue": [100]}), code_executed="df.groupby(...)")
    answer = compose_response("anything", result, llm)
    assert "region" in answer and "100" in answer


def test_compose_response_fallback_truncates_wide_tables_with_a_note():
    """A wide result must not dump every column into the persisted answer
    text (unreadable once repeated in chat history) — it should preview a
    few columns/rows and say so, while the full result is shown separately."""
    llm = make_llm(raises=True)
    wide_df = pd.DataFrame({f"col{i}": range(20) for i in range(10)})
    result = ExecutionResult(op_type="describe", result_df=wide_df, code_executed="df.describe()")
    answer = compose_response("anything", result, llm)
    assert "first 6 of 10 columns" in answer
    assert "first 5 of 20 rows" in answer
    assert "col9" not in answer  # beyond the 6-column preview cap


def test_compose_response_fallback_no_truncation_note_for_small_result():
    llm = make_llm(raises=True)
    small_df = pd.DataFrame({"region": ["A", "B"], "revenue": [100, 200]})
    result = ExecutionResult(op_type="groupby", result_df=small_df, code_executed="df.groupby(...)")
    answer = compose_response("anything", result, llm)
    assert "showing" not in answer
