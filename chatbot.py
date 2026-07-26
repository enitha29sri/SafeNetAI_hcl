"""
chatbot.py

High-level chat orchestration for SafeNet AI.

Combines the system prompt, conversation memory, optional RAG context, and
the Gemini LLM client into a single `ChatBot` class that the Streamlit UI
(app.py) and services/chat_service.py call into. This is the main entry
point for the "AI Chat" and RAG-grounded conversation features.
"""

from __future__ import annotations

from typing import Optional

from models.llm import GeminiLLM, GeminiLLMError
from models.memory import ConversationMemory
from prompts import SAFENET_SYSTEM_PROMPT, build_rag_context_block
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatBotError(Exception):
    """Raised when the chatbot cannot produce a response."""


class ChatBot:
    """
    Orchestrates SafeNet AI's conversational chat experience.

    Responsibilities:
        - Maintain conversation memory across turns.
        - Inject the system persona and (optionally) RAG context.
        - Delegate text generation to GeminiLLM.
        - Fail gracefully with user-friendly error messages.
    """

    def __init__(
        self,
        llm: Optional[GeminiLLM] = None,
        memory: Optional[ConversationMemory] = None,
        max_turns: int = 12,
    ) -> None:
        """
        Args:
            llm: An existing GeminiLLM instance to reuse, or None to create one.
            memory: An existing ConversationMemory to reuse, or None to create one.
            max_turns: Passed to a newly created ConversationMemory (ignored if
                `memory` is supplied).

        Raises:
            ChatBotError: If the underlying LLM client fails to initialize.
        """
        try:
            self._llm = llm or GeminiLLM()
        except GeminiLLMError as exc:
            logger.error(f"ChatBot failed to initialize LLM: {exc}")
            raise ChatBotError(str(exc)) from exc

        self._memory = memory or ConversationMemory(max_turns=max_turns)

    @property
    def memory(self) -> ConversationMemory:
        """Expose the underlying conversation memory (e.g. for chat history UI)."""
        return self._memory

    def send_message(
        self,
        user_input: str,
        context_chunks: Optional[list[str]] = None,
    ) -> str:
        """
        Send a user message to Gemini and return the assistant's reply.

        Args:
            user_input: The raw text typed by the user.
            context_chunks: Optional RAG excerpts to ground the response in.

        Returns:
            The assistant's reply text.

        Raises:
            ChatBotError: If input is empty/invalid or generation fails.
        """
        cleaned_input = (user_input or "").strip()
        if not cleaned_input:
            raise ChatBotError("Please enter a message before sending.")

        system_content = SAFENET_SYSTEM_PROMPT
        rag_block = build_rag_context_block(context_chunks or [])
        if rag_block:
            system_content = f"{SAFENET_SYSTEM_PROMPT}\n\n{rag_block}"

        messages = [GeminiLLM.build_system_message(system_content)]
        messages.extend(self._memory.as_langchain_messages())
        messages.append(GeminiLLM.build_human_message(cleaned_input))

        try:
            reply = self._llm.generate(messages)
        except GeminiLLMError as exc:
            logger.error(f"ChatBot generation failed: {exc}")
            raise ChatBotError(str(exc)) from exc

        self._memory.add_user_message(cleaned_input)
        self._memory.add_ai_message(reply)

        return reply

    def reset(self) -> None:
        """Clear the current conversation memory, starting a fresh session."""
        self._memory.clear()
        logger.info("ChatBot session reset.")
