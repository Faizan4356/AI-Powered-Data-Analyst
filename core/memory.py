"""Conversational memory: a short rolling history of prior Q&A turns.

Passed into intent_parser/operation_planner as plain text context so a
follow-up like "now break that down by region" resolves against the
previous question's subject instead of being classified in isolation.
Deliberately capped and text-only — this must never become a channel for
smuggling the raw dataset to the LLM; it only ever carries prior questions,
their intents/op_types, and the already-composed natural-language answers.
"""
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    question: str
    intent_type: str
    op_type: str
    answer: str


@dataclass
class ConversationMemory:
    max_turns: int = 5
    turns: list = field(default_factory=list)

    def add(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_context(self) -> str:
        if not self.turns:
            return ""
        lines = ["Prior conversation turns in this session (most recent last):"]
        for t in self.turns:
            lines.append(f"- Q: \"{t.question}\" -> intent={t.intent_type}, op_type={t.op_type} -> A: \"{t.answer}\"")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
