import pandas as pd

from core.executor import ExecutionResult
from core.self_correcting import SelfCorrectingResult
from core.trust_score import compute_trust_score
from core.validation import OperationPlan


def outcome(plan, result, attempts=1, attempt_log=None):
    return SelfCorrectingResult(plan=plan, result=result, attempts=attempts, attempt_log=attempt_log or [])


def test_full_trust_for_clean_first_attempt():
    plan = OperationPlan(op_type="groupby", groupby=["region"], agg_func="sum", agg_column="revenue")
    result = ExecutionResult(op_type="groupby", result_df=pd.DataFrame({"region": ["A"], "revenue": [100]}), code_executed="...")
    trust = compute_trust_score(outcome(plan, result))
    assert trust.score == 100
    assert trust.label == "High"


def test_zero_trust_on_execution_error():
    plan = OperationPlan(op_type="aggregation")
    result = ExecutionResult(op_type="aggregation", result_df=pd.DataFrame(), code_executed="", error="Unknown column: x")
    trust = compute_trust_score(outcome(plan, result))
    assert trust.score == 0
    assert trust.label == "Failed"


def test_medium_trust_for_bare_fallback_plan():
    plan = OperationPlan(op_type="describe", columns=["a", "b"])
    result = ExecutionResult(op_type="describe", result_df=pd.DataFrame({"a": [1]}), code_executed="...")
    trust = compute_trust_score(outcome(plan, result))
    assert trust.score == 60
    assert trust.label == "Medium"


def test_high_self_corrected_trust_after_retry_with_real_plan():
    plan = OperationPlan(op_type="filter", filters=[{"column": "region", "op": "==", "value": "A"}], columns=["region"])
    result = ExecutionResult(op_type="filter", result_df=pd.DataFrame({"region": ["A"]}), code_executed="...")
    trust = compute_trust_score(outcome(plan, result, attempts=2))
    assert trust.score == 80
    assert "self-corrected" in trust.label.lower()


def test_low_trust_for_fallback_plan_after_multiple_retries():
    plan = OperationPlan(op_type="describe")
    result = ExecutionResult(op_type="describe", result_df=pd.DataFrame({"a": [1]}), code_executed="...")
    trust = compute_trust_score(outcome(plan, result, attempts=3))
    assert trust.score == 40
    assert trust.label == "Low"
