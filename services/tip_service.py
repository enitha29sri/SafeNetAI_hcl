"""
services/tip_service.py

Provides a deterministic "tip of the day" from a curated cybersecurity
tip bank, and persists which tip was shown on which calendar date so the
same tip is shown consistently all day and remains reviewable later.
"""

from __future__ import annotations

from datetime import date

from database import Database, DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)

TIP_BANK: list[str] = [
    "Use a unique password for every account - a password manager makes this painless.",
    "Enable two-factor authentication wherever it's offered, especially for email and banking.",
    "Before clicking a link in an unexpected message, hover over it (or long-press on mobile) "
    "to preview the real destination.",
    "Official organizations will never ask for your password or OTP over phone, email, or SMS.",
    "Keep your phone and apps updated - many updates patch security vulnerabilities.",
    "Review app permissions periodically and revoke anything a mobile app no longer needs.",
    "Be skeptical of urgency: scammers create time pressure so you act before you think.",
    "Back up important files regularly so ransomware or device loss can't cost you everything.",
    "Check for HTTPS and the correct domain name before entering login details on any site.",
    "Avoid logging into sensitive accounts over public Wi-Fi without a VPN.",
    "Lock your devices with a PIN, password, or biometric - not just a swipe.",
    "Think before you overshare on social media; scammers use public details to craft convincing scams.",
]


class TipService:
    """High-level service for retrieving and persisting the daily cyber tip."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def get_today_tip(self) -> str:
        """
        Return today's tip, deterministically selected by day-of-year so
        it's stable across reloads, and persist it for the day if not
        already recorded.

        Returns:
            The tip text for today. Falls back to a computed (but
            unpersisted) tip if the database is temporarily unavailable.
        """
        today = date.today()

        try:
            existing = self._db.get_tip_for_date(today)
            if existing:
                return existing
        except DatabaseError as exc:
            logger.warning(f"Could not read stored daily tip, falling back to computed tip: {exc}")

        index = today.toordinal() % len(TIP_BANK)
        tip = TIP_BANK[index]

        try:
            self._db.save_tip_for_date(today, tip)
        except DatabaseError as exc:
            logger.warning(f"Could not persist daily tip (continuing anyway): {exc}")

        return tip
