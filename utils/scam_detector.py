"""
utils/scam_detector.py

Rule-based heuristic detector for phishing / scam patterns in free-text
messages (SMS, email, chat). Runs entirely locally with no external calls,
so it works even without a Gemini API key and never sends the pasted
message anywhere.

This is intentionally heuristic rather than a trained classifier: it is
transparent (every flag is explainable), fast, and good enough to catch
the most common social-engineering patterns for a safety-education tool.
"""

from __future__ import annotations

import html
import re

from utils.risk_analyzer import RiskFlag, RiskResult, build_risk_result

# --- Pattern groups, each targeting one social-engineering technique ---

_URGENCY_PATTERNS = [
    r"\bact now\b", r"\bact immediately\b", r"\burgent(ly)?\b",
    r"\bwithin\s+(24|48)\s+hours\b", r"\bimmediate action required\b",
    r"\blimited time\b", r"\bexpires? (today|soon)\b", r"\blast chance\b",
]

_THREAT_PATTERNS = [
    r"\baccount (has been |will be )?(suspended|locked|blocked|deactivated)\b",
    r"\blegal action\b", r"\byour account will be closed\b",
    r"\bfailure to (comply|respond)\b",
]

_LURE_PATTERNS = [
    r"\byou('| )ve won\b", r"\bcongratulations\b[\w\s]{0,30}\b(winner|selected|prize)\b",
    r"\bclaim your (prize|reward|gift)\b", r"\bfree (gift|prize|money)\b",
    r"\blottery\b", r"\bcashback\b[\w\s]{0,20}\bclaim\b",
]

_CREDENTIAL_REQUEST_PATTERNS = [
    r"\bverify your (account|identity|password|details)\b",
    r"\benter your (password|pin|otp|cvv)\b",
    r"\bconfirm your (bank|card|account) details\b",
    r"\bshare your (otp|pin|password)\b",
    r"\bsocial security number\b", r"\baadhaar\b[\w\s]{0,20}\bnumber\b",
]

_GENERIC_GREETING_PATTERNS = [
    r"\bdear (customer|user|valued customer|member)\b",
    r"\bdear sir/?madam\b",
]

_SUSPICIOUS_LINK_PATTERNS = [
    r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # raw IP address links
    r"https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|ow\.ly)/\S+",  # shorteners
    r"https?://[^\s]+\.(tk|ml|ga|cf|gq)\b",  # frequently abused free TLDs
]

_IMPERSONATION_PATTERNS = [
    r"\b(paypal|amazon|netflix|apple|microsoft|bank of|hdfc|icici|sbi)\b[\w\s]{0,60}"
    r"(verify|update|confirm|suspend)",
]


def _find_matches(text: str, patterns: list[str]) -> list[str]:
    """Return distinct, HTML-escaped matched snippets for a group of regex patterns."""
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = html.escape(match.group(0).strip())
            if snippet and snippet not in matches:
                matches.append(snippet)
    return matches


def _excessive_caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase, ignoring short text."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


class ScamDetector:
    """
    Heuristic phishing/scam pattern detector for pasted text messages.

    Never logs, stores, or transmits the analyzed message - all matching
    happens in-process against local regex patterns.
    """

    def analyze(self, message: str) -> RiskResult:
        """
        Analyze a message for common scam/phishing indicators.

        Args:
            message: The raw text pasted by the user.

        Returns:
            A RiskResult summarizing the detected risk flags and overall
            score/level. An empty or whitespace-only message returns a
            zero-score LOW result with no flags.
        """
        text = (message or "").strip()
        flags: list[RiskFlag] = []

        if not text:
            return build_risk_result(flags)

        if urgency := _find_matches(text, _URGENCY_PATTERNS):
            flags.append(RiskFlag(
                label="Urgency pressure",
                description=f'Uses urgent language to rush your decision (e.g. "{urgency[0]}").',
                weight=15,
            ))

        if threats := _find_matches(text, _THREAT_PATTERNS):
            flags.append(RiskFlag(
                label="Threatening consequences",
                description=f'Threatens a negative outcome to pressure you (e.g. "{threats[0]}").',
                weight=20,
            ))

        if lures := _find_matches(text, _LURE_PATTERNS):
            flags.append(RiskFlag(
                label="Too-good-to-be-true offer",
                description=f'Promises a prize, winnings, or free money (e.g. "{lures[0]}").',
                weight=20,
            ))

        if creds := _find_matches(text, _CREDENTIAL_REQUEST_PATTERNS):
            flags.append(RiskFlag(
                label="Requests sensitive credentials",
                description=(
                    f'Asks you to enter or share sensitive information (e.g. "{creds[0]}"). '
                    "Legitimate organizations never ask for this over message."
                ),
                weight=30,
            ))

        if greetings := _find_matches(text, _GENERIC_GREETING_PATTERNS):
            flags.append(RiskFlag(
                label="Generic greeting",
                description=(
                    f'Uses a generic greeting instead of your name (e.g. "{greetings[0]}"), '
                    "common in mass-sent scam messages."
                ),
                weight=8,
            ))

        if _find_matches(text, _SUSPICIOUS_LINK_PATTERNS):
            flags.append(RiskFlag(
                label="Suspicious link pattern",
                description=(
                    "Contains a link using a raw IP address, URL shortener, or an "
                    "uncommon/free domain often abused for phishing."
                ),
                weight=25,
            ))

        if _find_matches(text, _IMPERSONATION_PATTERNS):
            flags.append(RiskFlag(
                label="Possible brand impersonation",
                description=(
                    "Mentions a well-known brand alongside a request to verify, update, "
                    "or confirm account details - a classic phishing pattern."
                ),
                weight=20,
            ))

        caps_ratio = _excessive_caps_ratio(text)
        if caps_ratio > 0.4:
            flags.append(RiskFlag(
                label="Excessive capitalization",
                description="Uses unusually heavy capitalization, a common spam/scam signal.",
                weight=8,
            ))

        exclamations = text.count("!")
        if exclamations >= 3:
            flags.append(RiskFlag(
                label="Excessive punctuation",
                description=(
                    f"Contains {exclamations} exclamation marks, often used to create false urgency."
                ),
                weight=5,
            ))

        return build_risk_result(flags)
