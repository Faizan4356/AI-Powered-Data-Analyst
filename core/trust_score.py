"""Explainable trust score per answer: turns the audit log's operation-plan
trace into a short, user-facing verdict on how much to trust a given answer.

Deterministic, derived only from facts already established by validation and
execution — never an LLM's self-assessment of its own answer.
"""
from dataclasses import dataclass

from core.executor import ExecutionResult
from core.self_correcting import SelfCorrectingResult
from core.validation import OperationPlan


@dataclass
class TrustScore:
    score: int  # 0-100
    label: str
    reason: str


def compute_trust_score(outcome: SelfCorrectingResult) -> TrustScore:
    plan, result = outcome.plan, outcome.result

    if result.error:
        return TrustScore(score=0, label="Failed", reason=f"Execution failed after {outcome.attempts} attempt(s): {result.error}")

    used_fallback_plan = _is_bare_fallback(plan)

    if used_fallback_plan and outcome.attempts > 1:
        return TrustScore(
            score=40,
            label="Low",
            reason=(
                f"The system could not validate a plan matching your question after {outcome.attempts} attempts "
                "and fell back to a generic data preview. The answer may not address your exact question."
            ),
        )

    if used_fallback_plan:
        return TrustScore(
            score=60,
            label="Medium",
            reason=(
                "The LLM's proposed plan could not be validated against the real schema (or the LLM was "
                "unavailable), so a safe generic preview was used instead. The result is real and reproducible, "
                "but may not directly answer your question."
            ),
        )

    if outcome.attempts > 1:
        return TrustScore(
            score=80,
            label="High (self-corrected)",
            reason=(
                f"The first plan didn't produce a usable result, so the system retried and succeeded on "
                f"attempt {outcome.attempts}. The final plan was validated against the real schema and executed cleanly."
            ),
        )

    return TrustScore(
        score=100,
        label="High",
        reason="The plan was validated against the real schema and executed without error on the first attempt.",
    )


def _is_bare_fallback(plan: OperationPlan) -> bool:
    return plan.op_type == "describe" and not plan.filters and not plan.groupby and not plan.agg_func
