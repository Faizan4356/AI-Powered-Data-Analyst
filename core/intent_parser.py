"""LLM step 1: classify the user's natural-language question into an intent.

This is classification only — no numbers are produced or interpreted here.
"""
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from core.llm_client import LLMClient, LLMUnavailableError, safe_json_parse
from core.memory import ConversationMemory

INTENT_TYPES = [
    "aggregation",
    "trend",
    "comparison",
    "forecast",
    "segmentation",
    "anomaly",
    "explanation",
]

SYSTEM_PROMPT = f"""You are an intent classifier for a data analysis tool.
Classify the user's question into exactly one of: {', '.join(INTENT_TYPES)}.
You may be given prior conversation turns for context — use them only to resolve
follow-up questions (e.g. "now break that down by region" refers to the previous
question's subject), never as a source of facts or numbers.
Respond with strict JSON only: {{"intent": "<one of the types>", "reasoning": "<one short sentence>"}}
Do not compute or state any numeric answer. You only classify."""


@dataclass
class Intent:
    intent_type: str
    reasoning: str
    raw_question: str


def parse_intent(question: str, llm: LLMClient, memory: Optional[ConversationMemory] = None) -> Intent:
    user_prompt = question
    if memory is not None:
        context = memory.as_context()
        if context:
            user_prompt = f"{context}\n\nCurrent question: {question}"
    try:
        resp = llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        data = safe_json_parse(resp.text)
        intent_type = data.get("intent", "aggregation")
        if intent_type not in INTENT_TYPES:
            intent_type = "aggregation"
        return Intent(intent_type=intent_type, reasoning=data.get("reasoning", ""), raw_question=question)
    except LLMUnavailableError as e:
        logger.warning(f"Intent parsing degraded (LLM unavailable): {e}")
        return Intent(intent_type="aggregation", reasoning="LLM unavailable — defaulted to aggregation", raw_question=question)
