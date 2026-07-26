"""
models/llm.py

Wraps the Google Gemini chat model via LangChain's ChatGoogleGenerativeAI,
providing a single, testable interface for text generation used throughout
SafeNet AI (direct chat, RAG-grounded answers, quiz generation, etc.)
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMError(Exception):
    """Raised when the Gemini LLM fails to initialize or generate a response."""


class GeminiLLM:
    """
    Thin, defensive wrapper around LangChain's ChatGoogleGenerativeAI.

    Centralizing model instantiation here means every other module
    (chatbot.py, rag.py, services/quiz_service.py, ...) gets a single,
    consistently configured client instead of re-reading settings or
    re-handling API errors independently.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 1024,
    ) -> None:
        """
        Initialize the Gemini chat model client.

        Args:
            model_name: Gemini model identifier. Defaults to settings value.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            max_output_tokens: Hard cap on generated tokens per response.

        Raises:
            GeminiLLMError: If the API key is missing or client init fails.
        """
        if not settings.gemini_api_key:
            raise GeminiLLMError(
                "GEMINI_API_KEY is not configured. Add it to your .env file "
                "(see .env.example)."
            )

        self._model_name = model_name or settings.gemini_model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        try:
            self._client = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=settings.gemini_api_key,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                convert_system_message_to_human=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a domain-specific error
            logger.error(f"Failed to initialize Gemini client: {exc}")
            raise GeminiLLMError(f"Could not initialize Gemini client: {exc}") from exc

        logger.info(f"GeminiLLM initialized with model '{self._model_name}'.")

    def generate(self, messages: list[BaseMessage]) -> str:
        """
        Generate a text response for a sequence of chat messages.

        Args:
            messages: Ordered list of LangChain message objects
                (SystemMessage, HumanMessage, AIMessage) forming the
                conversation to send to Gemini.

        Returns:
            The generated response text, stripped of leading/trailing whitespace.

        Raises:
            GeminiLLMError: If generation fails for any reason (network,
                quota, invalid input, empty response, etc.)
        """
        if not messages:
            raise GeminiLLMError("Cannot generate a response from an empty message list.")

        try:
            result = self._client.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Gemini generation failed: {exc}")
            raise GeminiLLMError(
                "SafeNet AI could not reach the Gemini service right now. "
                "Please try again in a moment."
            ) from exc

        content = getattr(result, "content", "")
        if not content or not isinstance(content, str):
            logger.warning("Gemini returned an empty or non-text response.")
            raise GeminiLLMError("Gemini returned an empty response.")

        return content.strip()

    @staticmethod
    def build_system_message(content: str) -> SystemMessage:
        """Wrap raw text in a LangChain SystemMessage."""
        return SystemMessage(content=content)

    @staticmethod
    def build_human_message(content: str) -> HumanMessage:
        """Wrap raw text in a LangChain HumanMessage."""
        return HumanMessage(content=content)

    @staticmethod
    def build_ai_message(content: str) -> AIMessage:
        """Wrap raw text in a LangChain AIMessage."""
        return AIMessage(content=content)
