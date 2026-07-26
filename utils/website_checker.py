"""
utils/website_checker.py

Rule-based heuristic advisor for website/URL safety. Parses a URL and
checks it against a set of well-known phishing/typosquatting indicators.

This performs NO live network requests (no DNS lookups, no fetching the
page, no contacting the target site whatsoever) - purely structural
analysis of the URL string itself, so it works offline and can never be
used to probe or interact with a potentially malicious destination.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from utils.risk_analyzer import RiskFlag, RiskResult, build_risk_result

_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan",
    "men", "date", "faith", "review", "trade", "accountant",
}

_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "buff.ly", "cutt.ly", "rebrand.ly",
}

_KNOWN_BRANDS = [
    "paypal", "amazon", "netflix", "apple", "microsoft", "google",
    "facebook", "instagram", "whatsapp", "bankofamerica", "hdfcbank",
    "icicibank", "sbi", "chase", "wellsfargo", "linkedin",
]

_SENSITIVE_KEYWORDS = ["login", "verify", "secure", "account", "update", "confirm", "signin"]


def _domain_parts(hostname: str) -> tuple[str, str]:
    """Split a hostname into (second-level label, tld) using a simple last-two-labels heuristic."""
    labels = hostname.split(".")
    if len(labels) < 2:
        return hostname, ""
    return labels[-2], labels[-1]


def _looks_like_ip(hostname: str) -> bool:
    """Whether the hostname is a raw IPv4 address rather than a domain name."""
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", hostname))


def _contains_lookalike_brand(hostname: str) -> str | None:
    """
    Check whether the hostname contains a known brand name (allowing common
    homoglyph substitutions like 0->o, 1->l) while its actual registrable
    domain doesn't match that brand - a classic typosquatting pattern.

    Returns:
        The matched brand name, or None if no lookalike pattern was found.
    """
    normalized = hostname.lower().replace("0", "o").replace("1", "l").replace("3", "e")
    domain, _ = _domain_parts(hostname)

    for brand in _KNOWN_BRANDS:
        if brand in normalized and domain.lower() != brand:
            return brand
    return None


class WebsiteCheckerError(Exception):
    """Raised when the provided input cannot be parsed as a URL at all."""


class WebsiteChecker:
    """
    Heuristic, fully offline URL/website safety advisor. Performs no live
    network requests - analyzes only the structure of the URL string.
    """

    def analyze(self, raw_url: str) -> RiskResult:
        """
        Analyze a URL for common phishing/typosquatting/structural red flags.

        Args:
            raw_url: The URL as typed or pasted by the user. A missing
                scheme (e.g. "example.com") is treated as "https://example.com".

        Returns:
            A RiskResult summarizing detected flags and overall score/level.

        Raises:
            WebsiteCheckerError: If the input is empty or has no parseable
                hostname at all.
        """
        cleaned = (raw_url or "").strip()
        if not cleaned:
            raise WebsiteCheckerError("Please enter a URL to analyze.")

        candidate = cleaned if "://" in cleaned else f"https://{cleaned}"

        try:
            parsed = urlparse(candidate)
        except ValueError as exc:
            raise WebsiteCheckerError(f"Could not parse that as a URL: {exc}") from exc

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise WebsiteCheckerError(
                "That doesn't look like a valid URL - a domain name is required."
            )

        flags: list[RiskFlag] = []

        if parsed.scheme == "http":
            flags.append(RiskFlag(
                label="No HTTPS encryption",
                description=(
                    "This site uses plain HTTP instead of HTTPS, so data sent to it "
                    "isn't encrypted in transit."
                ),
                weight=15,
            ))

        if _looks_like_ip(hostname):
            flags.append(RiskFlag(
                label="Raw IP address instead of a domain",
                description=(
                    "The link points directly to an IP address rather than a named "
                    "domain, a pattern rarely used by legitimate sites."
                ),
                weight=30,
            ))

        _, tld = _domain_parts(hostname)
        if tld in _SUSPICIOUS_TLDS:
            flags.append(RiskFlag(
                label="Uncommon/free top-level domain",
                description=(
                    f'Uses the ".{tld}" domain extension, which is free or cheap to '
                    "register and frequently abused for phishing sites."
                ),
                weight=15,
            ))

        if hostname in _URL_SHORTENERS or any(
            hostname.endswith("." + s) for s in _URL_SHORTENERS
        ):
            flags.append(RiskFlag(
                label="URL shortener",
                description=(
                    "This is a shortened link - the real destination is hidden until "
                    "you click it, which scammers exploit to disguise malicious sites."
                ),
                weight=20,
            ))

        if not _looks_like_ip(hostname) and hostname.count(".") >= 3:
            flags.append(RiskFlag(
                label="Excessive subdomains",
                description=(
                    f'The domain "{hostname}" has an unusually deep subdomain structure, '
                    "sometimes used to make a fake domain resemble a trusted one."
                ),
                weight=12,
            ))

        if hostname.count("-") >= 2:
            flags.append(RiskFlag(
                label="Multiple hyphens in domain",
                description=(
                    "The domain contains several hyphens, a pattern common in "
                    "hastily-registered lookalike domains."
                ),
                weight=10,
            ))

        lookalike = _contains_lookalike_brand(hostname)
        if lookalike:
            flags.append(RiskFlag(
                label="Possible brand impersonation",
                description=(
                    f'The domain resembles the brand "{lookalike}" but does not match '
                    "its official domain - a common typosquatting technique."
                ),
                weight=30,
            ))

        if "@" in candidate:
            flags.append(RiskFlag(
                label='"@" symbol in URL',
                description=(
                    "URLs containing '@' can trick browsers into showing a trusted "
                    "domain while actually navigating elsewhere."
                ),
                weight=25,
            ))

        if parsed.port is not None and parsed.port not in (80, 443):
            flags.append(RiskFlag(
                label="Unusual port number",
                description=(
                    f"Connects on port {parsed.port}, which is unusual for a normal "
                    "website and sometimes used to evade filters."
                ),
                weight=10,
            ))

        path_and_query = f"{parsed.path} {parsed.query}".lower()
        matched_keywords = [kw for kw in _SENSITIVE_KEYWORDS if kw in path_and_query]
        if matched_keywords and (lookalike or tld in _SUSPICIOUS_TLDS or _looks_like_ip(hostname)):
            flags.append(RiskFlag(
                label="Sensitive action requested on a suspicious domain",
                description=(
                    f'The URL path references "{matched_keywords[0]}" while the domain '
                    "itself already shows other risk signals - a common phishing combination."
                ),
                weight=15,
            ))

        if len(candidate) > 120:
            flags.append(RiskFlag(
                label="Unusually long URL",
                description=(
                    "Very long URLs are sometimes used to hide the true destination or "
                    "obscure suspicious parameters."
                ),
                weight=8,
            ))

        return build_risk_result(flags)
