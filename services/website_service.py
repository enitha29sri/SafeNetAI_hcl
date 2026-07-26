"""
services/website_service.py

Orchestrates the Website Safety Advisor feature: runs the local heuristic
WebsiteChecker (no live requests, no external calls) and optionally asks
Gemini for a plain-language explanation grounded in the detected flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.llm import GeminiLLM, GeminiLLMError
from utils.logger import get_logger
from utils.risk_analyzer import RiskResult
from utils.website_checker import WebsiteChecker

logger = get_logger(__name__)

_WEBSITE_EXPLAINER_SYSTEM_PROMPT = """\
You are SafeNet AI's Website Safety Advisor. You will be given a URL a user
is considering visiting, along with a list of heuristic risk flags already
detected in its structure. Write a short, plain-language explanation
(3-5 sentences) of why this URL is or isn't likely risky, referencing the
specific flags where relevant, and end with one clear, practical
recommendation. Do not repeat the flags verbatim as a list - write
naturally. Keep a calm, reassuring tone even for high-risk URLs. Note that
this analysis is structural only (no live page content was checked).
"""


@dataclass
class WebsiteAnalysisResult:
    """Combined output of the heuristic checker and optional AI explanation."""

    url: str
    risk_result: RiskResult
    ai_explanation: Optional[str]
    ai_available: bool


class WebsiteAnalysisService:
    """
    High-level service for analyzing a URL for structural safety risk.

    Always runs the local heuristic checker. Optionally augments the
    result with a Gemini-generated plain-language explanation, degrading
    gracefully to heuristic-only output if the LLM is unavailable.
    """

    def __init__(self, llm: Optional[GeminiLLM] = None) -> None:
        self._checker = WebsiteChecker()
        self._llm: Optional[GeminiLLM] = llm
        self._llm_init_attempted = llm is not None

    def _get_llm(self) -> Optional[GeminiLLM]:
        """
        Lazily initialize the Gemini client, remembering failures so we
        don't retry a broken configuration on every single analysis call.
        """
        if self._llm is not None or self._llm_init_attempted:
            return self._llm

        self._llm_init_attempted = True
        try:
            self._llm = GeminiLLM(temperature=0.3, max_output_tokens=300)
        except GeminiLLMError as exc:
            logger.warning(f"Website advisor running without AI explanation: {exc}")
            self._llm = None

        return self._llm

    def analyze(self, url: str, use_ai_explanation: bool = True) -> WebsiteAnalysisResult:
        """
        Analyze a URL for structural safety red flags.

        Args:
            url: The URL as typed/pasted by the user.
            use_ai_explanation: Whether to also request a Gemini-generated
                plain-language explanation.

        Returns:
            A WebsiteAnalysisResult with the heuristic RiskResult and,
            if available and requested, an AI-generated explanation.

        Raises:
            WebsiteCheckerError: If the URL cannot be parsed at all
                (propagated from WebsiteChecker so the caller can show a
                clear "invalid URL" message).
        """
        risk_result = self._checker.analyze(url)
        cleaned_url = url.strip()

        if not use_ai_explanation:
            return WebsiteAnalysisResult(
                url=cleaned_url, risk_result=risk_result, ai_explanation=None, ai_available=False
            )

        llm = self._get_llm()
        if llm is None:
            return WebsiteAnalysisResult(
                url=cleaned_url, risk_result=risk_result, ai_explanation=None, ai_available=False
            )

        flags_summary = "; ".join(
            f"{flag.label}: {flag.description}" for flag in risk_result.flags
        ) or "No heuristic flags were triggered."

        prompt = (
            f"URL: {cleaned_url}\n"
            f"Detected risk score: {risk_result.score}/100 ({risk_result.level.value})\n"
            f"Detected flags: {flags_summary}"
        )

        try:
            explanation = llm.generate([
                GeminiLLM.build_system_message(_WEBSITE_EXPLAINER_SYSTEM_PROMPT),
                GeminiLLM.build_human_message(prompt),
            ])
        except GeminiLLMError as exc:
            logger.warning(f"AI explanation generation failed, returning heuristic-only result: {exc}")
            return WebsiteAnalysisResult(
                url=cleaned_url, risk_result=risk_result, ai_explanation=None, ai_available=False
            )

        return WebsiteAnalysisResult(
            url=cleaned_url, risk_result=risk_result, ai_explanation=explanation, ai_available=True
        )
