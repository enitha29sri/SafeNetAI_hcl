"""
services/scam_service.py

Orchestrates the Scam Message Analyzer feature: runs the local heuristic
ScamDetector first (always available, no API key needed), then optionally
asks Gemini for a plain-language explanation grounded in the detected
flags. If Gemini is unavailable, the service still returns a complete,
useful result from the heuristic layer alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.llm import GeminiLLM, GeminiLLMError
from utils.logger import get_logger
from utils.risk_analyzer import RiskResult
from utils.scam_detector import ScamDetector

logger = get_logger(__name__)

_SCAM_EXPLAINER_SYSTEM_PROMPT = """\
You are SafeNet AI's Scam Message Analyzer. You will be given a message a
user received, along with a list of heuristic risk flags already detected
in it. Write a short, plain-language explanation (3-5 sentences) of why
this message is or isn't likely a scam, referencing the specific flags
where relevant, and end with one clear, practical recommendation for what
the user should do next. Do not repeat the flags verbatim as a list - write
naturally. Keep a calm, reassuring tone even for high-risk messages.
"""


@dataclass
class ScamAnalysisResult:
    """Combined output of the heuristic detector and optional AI explanation."""

    risk_result: RiskResult
    ai_explanation: Optional[str]
    ai_available: bool


class ScamAnalysisService:
    """
    High-level service for analyzing a pasted message for scam/phishing risk.

    Always runs the local heuristic detector. Optionally augments the
    result with a Gemini-generated plain-language explanation, degrading
    gracefully to heuristic-only output if the LLM is unavailable.
    """

    def __init__(self, llm: Optional[GeminiLLM] = None) -> None:
        self._detector = ScamDetector()
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
            logger.warning(f"Scam analyzer running without AI explanation: {exc}")
            self._llm = None

        return self._llm

    def analyze(self, message: str, use_ai_explanation: bool = True) -> ScamAnalysisResult:
        """
        Analyze a pasted message for scam/phishing indicators.

        Args:
            message: The raw text the user pasted.
            use_ai_explanation: Whether to also request a Gemini-generated
                plain-language explanation.

        Returns:
            A ScamAnalysisResult with the heuristic RiskResult and, if
            available and requested, an AI-generated explanation.
        """
        risk_result = self._detector.analyze(message)

        cleaned = (message or "").strip()
        if not cleaned or not use_ai_explanation:
            return ScamAnalysisResult(risk_result=risk_result, ai_explanation=None, ai_available=False)

        llm = self._get_llm()
        if llm is None:
            return ScamAnalysisResult(risk_result=risk_result, ai_explanation=None, ai_available=False)

        flags_summary = "; ".join(
            f"{flag.label}: {flag.description}" for flag in risk_result.flags
        ) or "No heuristic flags were triggered."

        prompt = (
            f'Message:\n"""\n{cleaned}\n"""\n\n'
            f"Detected risk score: {risk_result.score}/100 ({risk_result.level.value})\n"
            f"Detected flags: {flags_summary}"
        )

        try:
            explanation = llm.generate([
                GeminiLLM.build_system_message(_SCAM_EXPLAINER_SYSTEM_PROMPT),
                GeminiLLM.build_human_message(prompt),
            ])
        except GeminiLLMError as exc:
            logger.warning(f"AI explanation generation failed, returning heuristic-only result: {exc}")
            return ScamAnalysisResult(risk_result=risk_result, ai_explanation=None, ai_available=False)

        return ScamAnalysisResult(risk_result=risk_result, ai_explanation=explanation, ai_available=True)
