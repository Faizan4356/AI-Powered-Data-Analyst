"""Thin LLM client abstraction. Supports Groq (hosted) and Ollama (local).

This module is the ONLY place that talks to an LLM. It exposes a single
`complete(system, user, json_mode)` call used by intent_parser, operation_planner,
and response_composer. It never performs or checks numeric computation itself.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger


@dataclass
class LLMResponse:
    text: str
    latency_ms: float
    tokens_estimate: int
    backend: str


class LLMUnavailableError(Exception):
    """Raised when no LLM backend can be reached. Callers must degrade gracefully."""


class LLMClient:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or os.environ.get("LLM_BACKEND", "groq")
        # Every call (success or failure) is logged here for the LLM Usage
        # dashboard — token/latency transparency, not a mystery line item.
        self.usage_log: list = []

    def complete(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> LLMResponse:
        start = time.time()
        try:
            if self.backend == "groq":
                text = self._complete_groq(system, user, json_mode, temperature)
            elif self.backend == "ollama":
                text = self._complete_ollama(system, user, json_mode, temperature)
            else:
                raise LLMUnavailableError(f"Unknown LLM_BACKEND '{self.backend}'")
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.usage_log.append({"backend": self.backend, "latency_ms": round(latency_ms, 1), "tokens_estimate": 0, "success": False})
            logger.warning(f"LLM call failed on backend={self.backend}: {e}")
            raise LLMUnavailableError(str(e)) from e

        latency_ms = (time.time() - start) * 1000
        tokens_estimate = len(system.split()) + len(user.split()) + len(text.split())
        self.usage_log.append({"backend": self.backend, "latency_ms": round(latency_ms, 1), "tokens_estimate": tokens_estimate, "success": True})
        return LLMResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_estimate=tokens_estimate,
            backend=self.backend,
        )

    def _complete_groq(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY not set")
        client = Groq(api_key=api_key)
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content

    def _complete_ollama(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        payload = {
            "model": model,
            "prompt": f"SYSTEM: {system}\n\nUSER: {user}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"]


def safe_json_parse(text: str) -> dict:
    """Parse LLM JSON output defensively (models sometimes wrap in markdown fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw: {text[:500]}")
