"""LLM step 2: turn (intent, question, schema) into a structured operation plan.

The LLM sees ONLY column names/dtypes and a small sample of values — never
full data. Output is strict JSON, immediately re-validated by validation.py
before anything executes. The LLM never sees or produces final numbers here.
"""
import json
from typing import Optional

import pandas as pd
from loguru import logger

from core.intent_parser import Intent
from core.llm_client import LLMClient, LLMUnavailableError, safe_json_parse
from core.memory import ConversationMemory
from core.validation import OperationPlan, ValidationError, validate_plan

SYSTEM_PROMPT = """You are an operation planner for a data analysis tool.
Given a user's question, its classified intent, and a dataset schema, output a
structured JSON operation plan describing which real computation to run.
You never compute the answer yourself — you only describe the operation.
You may be given prior conversation turns — use them only to resolve follow-up
questions (e.g. carry over a previous groupby/filter when the user says "now
break that down by X" or "same but for last year"), never as a source of numbers.

Output strict JSON with this shape:
{
  "op_type": "aggregation | trend | comparison | groupby | filter | describe",
  "columns": ["col1", ...],
  "filters": [{"column": "col", "op": "== | != | > | >= | < | <= | in | not in | contains", "value": ...}],
  "groupby": ["col1", ...],
  "agg_func": "sum | mean | median | min | max | count | nunique | std | var | null",
  "agg_column": "col or null",
  "sort_by": "col or null",
  "sort_desc": true,
  "limit": 100
}
Only use column names that appear in the provided schema. Do not invent columns."""


def build_schema_context(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().unique()[:3].tolist()
        lines.append(f"- {col} ({dtype}), sample values: {sample}")
    return "\n".join(lines)


def plan_operation(
    intent: Intent,
    df: pd.DataFrame,
    llm: LLMClient,
    memory: Optional[ConversationMemory] = None,
    error_context: Optional[str] = None,
) -> OperationPlan:
    schema = build_schema_context(df)
    context = memory.as_context() if memory is not None else ""
    context_block = f"{context}\n\n" if context else ""
    retry_block = (
        f"Your previous plan for this question failed with: \"{error_context}\". "
        f"Propose a corrected plan that avoids this problem.\n\n"
        if error_context
        else ""
    )
    user_prompt = (
        f"{context_block}{retry_block}"
        f"Question: {intent.raw_question}\n"
        f"Classified intent: {intent.intent_type}\n\n"
        f"Dataset schema:\n{schema}\n\n"
        f"Row count: {len(df)}"
    )
    try:
        resp = llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        plan_dict = safe_json_parse(resp.text)
    except LLMUnavailableError as e:
        logger.warning(f"Operation planning degraded (LLM unavailable): {e}")
        plan_dict = _fallback_plan(df)
    except ValueError as e:
        logger.warning(f"Planner returned invalid JSON, using fallback: {e}")
        plan_dict = _fallback_plan(df)

    try:
        return validate_plan(plan_dict, df)
    except ValidationError as e:
        logger.warning(f"Plan failed validation ({e}); falling back to describe")
        return validate_plan(_fallback_plan(df), df)


def _fallback_plan(df: pd.DataFrame) -> dict:
    """Manual query building when the LLM is unavailable or returns garbage.
    Includes every column — dropping columns here would silently hide the
    exact data (e.g. a revenue column) the user asked about."""
    return {"op_type": "describe", "columns": list(df.columns), "filters": [], "groupby": [], "limit": 100}
