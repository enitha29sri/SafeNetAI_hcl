"""
utils/risk_analyzer.py

Shared risk-scoring primitives used across SafeNet AI's analysis tools
(Scam Analyzer, Website Safety Advisor, and later checkers). Centralizing
the RiskLevel enum and RiskResult dataclass here keeps scoring semantics
and UI badge styling consistent across every tool instead of each one
inventing its own thresholds and labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    """Overall risk classification shared by all SafeNet AI analyzers."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class RiskFlag:
    """A single detected risk indicator within analyzed content."""

    label: str
    description: str
    weight: int  # contribution to the overall risk score (0-100 scale)


@dataclass
class RiskResult:
    """
    The outcome of a heuristic risk analysis: a 0-100 score, a categorical
    level derived from that score, and the individual flags that produced it.
    """

    score: int
    level: RiskLevel
    flags: list[RiskFlag] = field(default_factory=list)

    @property
    def color(self) -> str:
        """CSS variable reference matching this risk level, for badge text/border color."""
        return {
            RiskLevel.LOW: "var(--accent-safe)",
            RiskLevel.MEDIUM: "var(--accent-amber)",
            RiskLevel.HIGH: "var(--accent-amber)",
            RiskLevel.CRITICAL: "var(--accent-danger)",
        }[self.level]

    @property
    def background(self) -> str:
        """Dim CSS variable reference matching this risk level, for badge background."""
        return {
            RiskLevel.LOW: "var(--accent-safe-dim)",
            RiskLevel.MEDIUM: "var(--accent-amber-dim)",
            RiskLevel.HIGH: "var(--accent-amber-dim)",
            RiskLevel.CRITICAL: "var(--accent-danger-dim)",
        }[self.level]


def score_to_level(score: int) -> RiskLevel:
    """
    Convert a 0-100 risk score into a categorical RiskLevel.

    Args:
        score: A risk score; clamped internally to [0, 100].

    Returns:
        The corresponding RiskLevel.
    """
    clamped = max(0, min(100, score))
    if clamped < 25:
        return RiskLevel.LOW
    if clamped < 50:
        return RiskLevel.MEDIUM
    if clamped < 75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def build_risk_result(flags: list[RiskFlag], base_score: int = 0) -> RiskResult:
    """
    Aggregate a list of risk flags into a final RiskResult.

    Args:
        flags: Individual risk indicators detected in the content.
        base_score: A starting score before flag weights are added (usually 0).

    Returns:
        A RiskResult with the summed, clamped score and derived level.
    """
    total = base_score + sum(flag.weight for flag in flags)
    total = max(0, min(100, total))
    return RiskResult(score=total, level=score_to_level(total), flags=flags)
