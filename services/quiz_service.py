"""
services/quiz_service.py

Interactive cybersecurity quiz engine: a static question bank, question
selection, and persistence of quiz attempts via the SQLite Database layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from database import Database, DatabaseError, QuizScoreEntry
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuizQuestion:
    """A single multiple-choice cybersecurity quiz question."""

    question: str
    options: list[str]
    correct_index: int
    explanation: str


QUESTION_BANK: list[QuizQuestion] = [
    QuizQuestion(
        question="Which of these is the strongest password?",
        options=["password123", "Tr0ub4dor&3", "J7#mK9!qLp2$", "qwerty"],
        correct_index=2,
        explanation=(
            "Longer, random combinations of characters resist brute-force and dictionary "
            "attacks far better than predictable substitutions or common words."
        ),
    ),
    QuizQuestion(
        question="A message urgently asks you to 'verify your account' by clicking a link. "
                 "What should you do first?",
        options=[
            "Click the link immediately to avoid losing access",
            "Reply with your password to confirm",
            "Independently navigate to the official site instead of clicking the link",
            "Forward it to friends to warn them",
        ],
        correct_index=2,
        explanation=(
            "Typing the official address yourself (or using a bookmark) avoids phishing "
            "links entirely, regardless of how convincing the message looks."
        ),
    ),
    QuizQuestion(
        question="What does two-factor authentication (2FA) primarily protect against?",
        options=[
            "Slow internet speeds",
            "Someone logging in with just your stolen password",
            "Computer viruses",
            "Running out of storage space",
        ],
        correct_index=1,
        explanation=(
            "2FA requires a second proof of identity (like a code or app approval), so a "
            "stolen password alone usually isn't enough to log in."
        ),
    ),
    QuizQuestion(
        question="Which URL pattern is the biggest red flag?",
        options=[
            "https://www.amazon.com/orders",
            "http://192.168.4.12/amazon-login",
            "https://accounts.google.com/signin",
            "https://en.wikipedia.org/wiki/Security",
        ],
        correct_index=1,
        explanation=(
            "A raw IP address combined with a brand name in the path is a classic phishing "
            "pattern — legitimate sites almost never operate this way."
        ),
    ),
    QuizQuestion(
        question="An app requests access to your contacts, but it's a simple flashlight app. "
                 "What should you do?",
        options=[
            "Grant it - all apps need broad permissions",
            "Deny it - a flashlight has no legitimate need for your contacts",
            "Ignore the request; it won't matter",
            "Uninstall your contacts app instead",
        ],
        correct_index=1,
        explanation=(
            "Permissions should match what an app actually needs to function. A flashlight "
            "app has no legitimate reason to read your contacts."
        ),
    ),
    QuizQuestion(
        question="What is 'typosquatting'?",
        options=[
            "A typing technique for faster passwords",
            "Registering a domain that closely resembles a popular brand's real domain",
            "A method to compress website files",
            "A firewall configuration setting",
        ],
        correct_index=1,
        explanation=(
            "Typosquatters register lookalike domains (e.g. 'paypa1.com') hoping users "
            "mistype or don't notice the difference, then use them to phish credentials."
        ),
    ),
    QuizQuestion(
        question="You receive an email saying you've won a lottery you never entered. "
                 "What's the safest response?",
        options=[
            "Reply asking how to claim the prize",
            "Delete or report it - you can't win a lottery you never entered",
            "Send your bank details to receive the winnings",
            "Call the number provided immediately",
        ],
        correct_index=1,
        explanation=(
            "Unsolicited prize notifications for contests you never entered are a classic "
            "scam lure designed to extract personal or financial information."
        ),
    ),
    QuizQuestion(
        question="Why is public Wi-Fi risky for sensitive tasks like banking?",
        options=[
            "It's always slower than home Wi-Fi",
            "Traffic can potentially be intercepted by others on the same network",
            "Public Wi-Fi blocks banking apps automatically",
            "It uses more battery",
        ],
        correct_index=1,
        explanation=(
            "Unsecured or shared networks make it easier for attackers to intercept traffic; "
            "using a VPN or your mobile data is safer for sensitive tasks."
        ),
    ),
]


class QuizService:
    """High-level service for running and scoring the cybersecurity quiz."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def get_random_questions(self, count: int = 5) -> list[QuizQuestion]:
        """
        Select a random subset of questions for one quiz attempt.

        Args:
            count: Number of questions to include, capped at the bank size.

        Returns:
            A shuffled list of QuizQuestion objects.
        """
        count = min(count, len(QUESTION_BANK))
        return random.sample(QUESTION_BANK, count)

    def save_attempt(self, session_id: str, score: int, total: int) -> None:
        """
        Persist a completed quiz attempt. Logs a warning instead of raising
        if persistence fails, so a database hiccup never blocks the user
        from seeing their result.

        Args:
            session_id: The anonymous session identifier.
            score: Number of correctly answered questions.
            total: Total number of questions in the attempt.
        """
        try:
            self._db.add_quiz_score(session_id, score, total)
        except DatabaseError as exc:
            logger.warning(f"Could not save quiz score (continuing anyway): {exc}")

    def get_history(self, session_id: str) -> list[QuizScoreEntry]:
        """Return this session's past quiz attempts, most recent first."""
        try:
            return self._db.get_quiz_scores(session_id)
        except DatabaseError as exc:
            logger.warning(f"Could not load quiz history: {exc}")
            return []
