"""LLM step 3: turn an already-computed ExecutionResult into natural language.

CRITICAL: this is given ONLY the computed result (a small table/scalar), never
the raw dataset. The system prompt explicitly forbids inventing or adjusting
numbers. If the LLM is unavailable, a deterministic templated fallback is used
so the app degrades gracefully instead of failing.
"""
import pandas as pd
from loguru import logger

from core.executor import ExecutionResult
from core.llm_client import LLMClient, LLMUnavailableError

SYSTEM_PROMPT = """You are a data analysis assistant. You will be given the ORIGINAL
QUESTION and a RESULT TABLE that was already computed by real code (Pandas/SQL).

Rules you must never break:
1. Only reference numbers that literally appear in the RESULT TABLE below.
2. Never invent, estimate, round differently, recompute, or "correct" any number.
3. If the result is empty or doesn't answer the question, say so plainly.
4. Be concise: 2-4 sentences, plain business language, no code.
"""


def compose_response(question: str, result: ExecutionResult, llm: LLMClient) -> str:
    if result.error:
        return f"The query could not be executed: {result.error}"

    table_str = _result_to_text(result)

    try:
        user_prompt = f"Original question: {question}\n\nRESULT TABLE (the only source of truth for numbers):\n{table_str}"
        resp = llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=False, temperature=0.2)
        return resp.text.strip()
    except LLMUnavailableError as e:
        logger.warning(f"Response composition degraded (LLM unavailable): {e}")
        return _templated_fallback(result)


def _result_to_text(result: ExecutionResult) -> str:
    """Full table, used only as private context sent to the LLM — never shown
    to the user directly, so no size cap needed beyond a sane row limit."""
    if result.scalar_value is not None:
        return f"{result.op_type} = {result.scalar_value}"
    if result.result_df.empty:
        return "(empty result)"
    return result.result_df.head(50).to_markdown(index=False)


MAX_FALLBACK_PREVIEW_COLS = 6
MAX_FALLBACK_PREVIEW_ROWS = 5


def _templated_fallback(result: ExecutionResult) -> str:
    """User-facing text when the LLM is unavailable. Deliberately a short
    preview, not a full table dump — the complete result is always rendered
    separately as a real dataframe; this text also gets persisted verbatim
    into chat history, where a full wide table would be unreadable."""
    if result.scalar_value is not None:
        return f"Computed result: {result.scalar_value}"

    df = result.result_df
    if df.empty:
        if df.columns.empty:
            return "No rows matched this query, and the join/upload produced no columns either — check the data upstream."
        return "No rows matched this query."

    n_rows, n_cols = df.shape
    preview = df.iloc[:MAX_FALLBACK_PREVIEW_ROWS, :MAX_FALLBACK_PREVIEW_COLS]
    preview_note = []
    if n_cols > MAX_FALLBACK_PREVIEW_COLS:
        preview_note.append(f"first {MAX_FALLBACK_PREVIEW_COLS} of {n_cols} columns")
    if n_rows > MAX_FALLBACK_PREVIEW_ROWS:
        preview_note.append(f"first {MAX_FALLBACK_PREVIEW_ROWS} of {n_rows} rows")
    note = f" (showing {', '.join(preview_note)} — full result shown below)" if preview_note else ""

    return f"Computed result{note} — LLM explanation unavailable:\n\n{preview.to_markdown(index=False)}"
