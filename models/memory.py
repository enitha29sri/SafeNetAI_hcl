"""
models/memory.py

Conversation memory management for SafeNet AI.

Keeps a bounded, ordered history of the current chat session so the LLM
has conversational context without the prompt growing unbounded. Backed
by simple LangChain message objects so it plugs directly into
models/llm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single user/assistant message, with a timestamp for display/export."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationMemory:
    """
    Bounded, in-memory conversation history for a single chat session.

    Args:
        max_turns: Maximum number of (user, assistant) turn pairs to retain.
            Older turns are dropped once this limit is exceeded, keeping
            the prompt sent to Gemini within a reasonable size.
    """

    def __init__(self, max_turns: int = 12) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self._max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def add_user_message(self, content: str) -> None:
        """Record a message sent by the user."""
        self._turns.append(ConversationTurn(role="user", content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        """Record a message returned by the assistant."""
        self._turns.append(ConversationTurn(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest turns once the buffer exceeds 2 * max_turns entries."""
        limit = self._max_turns * 2
        if len(self._turns) > limit:
            overflow = len(self._turns) - limit
            self._turns = self._turns[overflow:]
            logger.debug(f"Trimmed {overflow} old conversation turn(s) from memory.")

    def as_langchain_messages(self) -> list[BaseMessage]:
        """
        Convert stored turns into LangChain message objects, ready to be
        passed to GeminiLLM.generate() alongside a system prompt.
        """
        messages: list[BaseMessage] = []
        for turn in self._turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages

    def get_history(self) -> list[ConversationTurn]:
        """Return the full stored history, oldest first, for display/export."""
        return list(self._turns)

    def clear(self) -> None:
        """Wipe all stored conversation turns."""
        self._turns.clear()
        logger.info("Conversation memory cleared.")

    def __len__(self) -> int:
        return len(self._turns)
