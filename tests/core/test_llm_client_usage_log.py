"""LLM cost/usage transparency: every call, success or failure, must be
logged with backend, latency, and token estimate — this backs the LLM Usage
dashboard so token spend is never a hidden line item.
"""
from unittest.mock import patch

import pytest

from core.llm_client import LLMClient, LLMUnavailableError


def test_successful_call_is_logged():
    llm = LLMClient(backend="groq")
    with patch.object(llm, "_complete_groq", return_value="hello world"):
        llm.complete("system prompt", "user prompt")
    assert len(llm.usage_log) == 1
    entry = llm.usage_log[0]
    assert entry["success"] is True
    assert entry["backend"] == "groq"
    assert entry["tokens_estimate"] > 0
    assert entry["latency_ms"] >= 0


def test_failed_call_is_logged_with_zero_tokens():
    llm = LLMClient(backend="groq")
    with patch.object(llm, "_complete_groq", side_effect=RuntimeError("network down")):
        with pytest.raises(LLMUnavailableError):
            llm.complete("system prompt", "user prompt")
    assert len(llm.usage_log) == 1
    entry = llm.usage_log[0]
    assert entry["success"] is False
    assert entry["tokens_estimate"] == 0


def test_usage_log_accumulates_across_calls():
    llm = LLMClient(backend="groq")
    with patch.object(llm, "_complete_groq", return_value="ok"):
        llm.complete("s1", "u1")
        llm.complete("s2", "u2")
    assert len(llm.usage_log) == 2
