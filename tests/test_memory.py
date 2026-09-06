"""Conversational memory: prior turns should reach the LLM prompt as text
context (never as raw data), and should be capped rather than growing forever.
"""
from unittest.mock import MagicMock

import pandas as pd

from core.intent_parser import Intent, parse_intent
from core.llm_client import LLMResponse
from core.memory import ConversationMemory, ConversationTurn
from core.operation_planner import plan_operation


def make_llm(text):
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text=text, latency_ms=1.0, tokens_estimate=1, backend="mock")
    return llm


def test_memory_caps_at_max_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add(ConversationTurn(question=f"q{i}", intent_type="aggregation", op_type="groupby", answer=f"a{i}"))
    assert len(memory.turns) == 2
    assert memory.turns[0].question == "q3"
    assert memory.turns[1].question == "q4"


def test_memory_as_context_empty_when_no_turns():
    assert ConversationMemory().as_context() == ""


def test_memory_clear():
    memory = ConversationMemory()
    memory.add(ConversationTurn(question="q", intent_type="aggregation", op_type="groupby", answer="a"))
    memory.clear()
    assert memory.turns == []


def test_parse_intent_includes_memory_in_prompt():
    llm = make_llm('{"intent": "trend", "reasoning": "follow-up"}')
    memory = ConversationMemory()
    memory.add(ConversationTurn(question="revenue by region", intent_type="aggregation", op_type="groupby", answer="Region A leads."))

    parse_intent("now break that down by month", llm, memory=memory)

    sent_user_prompt = llm.complete.call_args[0][1]
    assert "revenue by region" in sent_user_prompt
    assert "now break that down by month" in sent_user_prompt


def test_plan_operation_includes_memory_in_prompt():
    df = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    llm = make_llm('{"op_type": "groupby", "groupby": ["region"], "agg_func": "sum", "agg_column": "revenue"}')
    memory = ConversationMemory()
    memory.add(ConversationTurn(question="total revenue", intent_type="aggregation", op_type="aggregation", answer="30"))
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="now break that down by region")

    plan_operation(intent, df, llm, memory=memory)

    sent_user_prompt = llm.complete.call_args[0][1]
    assert "total revenue" in sent_user_prompt
    assert "break that down by region" in sent_user_prompt


def test_plan_operation_works_without_memory():
    df = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    llm = make_llm('{"op_type": "describe"}')
    intent = Intent(intent_type="aggregation", reasoning="", raw_question="describe the data")
    plan = plan_operation(intent, df, llm, memory=None)
    assert plan.op_type == "describe"
