"""Self-correcting execution loop: if a generated plan errors out or returns
an empty result, feed that error back to the operation planner and retry,
bounded to a small number of attempts — never a silent failure, never an
unbounded loop.

Every attempt still goes through the same validated plan -> deterministic
executor path as a single-shot query; only the *prompt* changes between
retries (it gains the previous failure as context). The LLM still never
computes a number — it only gets another chance to describe a better
operation.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.executor import ExecutionResult, execute_plan
from core.intent_parser import Intent
from core.llm_client import LLMClient
from core.memory import ConversationMemory
from core.operation_planner import plan_operation
from core.validation import OperationPlan

MAX_RETRIES = 2


@dataclass
class SelfCorrectingResult:
    plan: OperationPlan
    result: ExecutionResult
    attempts: int
    attempt_log: list = field(default_factory=list)  # [{"attempt", "op_type", "error"}]


def _is_unproductive(result: ExecutionResult) -> bool:
    if result.error:
        return True
    if result.scalar_value is not None:
        return False
    return result.result_df.empty


def plan_and_execute_with_retries(
    intent: Intent,
    df: pd.DataFrame,
    llm: LLMClient,
    memory: Optional[ConversationMemory] = None,
    max_retries: int = MAX_RETRIES,
) -> SelfCorrectingResult:
    error_context = None
    attempt_log = []
    plan = None
    result = None

    for attempt in range(max_retries + 1):
        plan = plan_operation(intent, df, llm, memory=memory, error_context=error_context)
        result = execute_plan(plan, df)
        attempt_log.append(
            {
                "attempt": attempt + 1,
                "op_type": plan.op_type,
                "error": result.error,
                "empty": result.scalar_value is None and result.result_df.empty,
            }
        )

        if not _is_unproductive(result):
            break

        error_context = result.error or "The query executed but returned no rows"

    return SelfCorrectingResult(plan=plan, result=result, attempts=len(attempt_log), attempt_log=attempt_log)
