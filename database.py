"""
database.py

SQLite persistence layer for SafeNet AI.

Provides a single `Database` class wrapping a local SQLite file for:
    - chat_history : persisted chat messages per (anonymous) session
    - quiz_scores  : quiz attempt results
    - feedback     : thumbs up/down + optional comments on assistant replies
    - settings     : simple key-value app settings
    - daily_tips   : which "tip of the day" was shown on which calendar date

All methods open a short-lived connection per call (the simplest safe
pattern under Streamlit's rerun model), validate input, and raise
DatabaseError on failure instead of leaking raw sqlite3 exceptions.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


@dataclass
class ChatHistoryEntry:
    """A single persisted chat message."""

    id: int
    session_id: str
    role: str
    content: str
    timestamp: datetime


@dataclass
class QuizScoreEntry:
    """A single persisted quiz attempt result."""

    id: int
    session_id: str
    score: int
    total: int
    timestamp: datetime


class Database:
    """SQLite-backed persistence layer for SafeNet AI."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Args:
            db_path: Path to the SQLite file. Defaults to settings.database_path.

        Raises:
            DatabaseError: If the schema cannot be created.
        """
        self._db_path = db_path or settings.database_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection, committing on success and rolling back on error."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Create all required tables if they don't already exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quiz_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_excerpt TEXT NOT NULL,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tip_date TEXT NOT NULL UNIQUE,
            tip_text TEXT NOT NULL
        );
        """
        try:
            with self._connect() as conn:
                conn.executescript(schema)
        except sqlite3.Error as exc:
            logger.error(f"Failed to initialize database schema: {exc}")
            raise DatabaseError(f"Failed to initialize database schema: {exc}") from exc

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Persist a single chat message."""
        if role not in ("user", "assistant"):
            raise DatabaseError(f"Invalid chat role '{role}'.")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_history (session_id, role, content, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, datetime.now().isoformat()),
                )
        except sqlite3.Error as exc:
            logger.error(f"Failed to save chat message: {exc}")
            raise DatabaseError(f"Failed to save chat message: {exc}") from exc

    def get_chat_history(self, session_id: str) -> list[ChatHistoryEntry]:
        """Return all messages for a session, oldest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error(f"Failed to fetch chat history: {exc}")
            raise DatabaseError(f"Failed to fetch chat history: {exc}") from exc

        return [
            ChatHistoryEntry(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    def clear_chat_history(self, session_id: str) -> None:
        """Delete all persisted messages for a session."""
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        except sqlite3.Error as exc:
            logger.error(f"Failed to clear chat history: {exc}")
            raise DatabaseError(f"Failed to clear chat history: {exc}") from exc

    def count_all_chat_messages(self) -> int:
        """Return the total number of chat messages across all sessions."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS c FROM chat_history").fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to count chat messages: {exc}") from exc
        return int(row["c"])

    def get_messages_per_day(self, days: int = 14) -> list[tuple[str, int]]:
        """Return (date, message_count) pairs for the last `days` days, oldest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS c
                    FROM chat_history
                    GROUP BY day
                    ORDER BY day DESC
                    LIMIT ?
                    """,
                    (days,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to aggregate chat messages: {exc}") from exc
        return [(row["day"], row["c"]) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Quiz scores
    # ------------------------------------------------------------------

    def add_quiz_score(self, session_id: str, score: int, total: int) -> None:
        """Persist one completed quiz attempt."""
        if total <= 0 or score < 0 or score > total:
            raise DatabaseError(f"Invalid quiz score: {score}/{total}.")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO quiz_scores (session_id, score, total, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, score, total, datetime.now().isoformat()),
                )
        except sqlite3.Error as exc:
            logger.error(f"Failed to save quiz score: {exc}")
            raise DatabaseError(f"Failed to save quiz score: {exc}") from exc

    def get_quiz_scores(self, session_id: Optional[str] = None) -> list[QuizScoreEntry]:
        """Return quiz attempts, optionally filtered to one session, most recent first."""
        query = "SELECT * FROM quiz_scores"
        params: tuple = ()
        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id DESC"

        try:
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to fetch quiz scores: {exc}") from exc

        return [
            QuizScoreEntry(
                id=row["id"],
                session_id=row["session_id"],
                score=row["score"],
                total=row["total"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    def get_average_quiz_percentage(self) -> Optional[float]:
        """Return the average quiz score across all attempts, as a percentage."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT SUM(score) AS s, SUM(total) AS t FROM quiz_scores"
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to compute average quiz score: {exc}") from exc

        if not row or not row["t"]:
            return None
        return round((row["s"] / row["t"]) * 100, 1)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def add_feedback(
        self, session_id: str, message_excerpt: str, rating: str, comment: Optional[str] = None
    ) -> None:
        """Persist a thumbs up/down (and optional comment) on an assistant message."""
        if rating not in ("up", "down"):
            raise DatabaseError(f"Invalid feedback rating '{rating}'.")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO feedback (session_id, message_excerpt, rating, comment, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, message_excerpt[:300], rating, comment, datetime.now().isoformat()),
                )
        except sqlite3.Error as exc:
            logger.error(f"Failed to save feedback: {exc}")
            raise DatabaseError(f"Failed to save feedback: {exc}") from exc

    def get_feedback_summary(self) -> dict[str, int]:
        """Return counts of {'up': n, 'down': n} feedback across all sessions."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT rating, COUNT(*) AS c FROM feedback GROUP BY rating"
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to summarize feedback: {exc}") from exc
        summary = {"up": 0, "down": 0}
        for row in rows:
            summary[row["rating"]] = row["c"]
        return summary

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Return a stored setting value, or `default` if unset."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to read setting '{key}': {exc}") from exc
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Create or update a stored setting value."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
        except sqlite3.Error as exc:
            logger.error(f"Failed to save setting '{key}': {exc}")
            raise DatabaseError(f"Failed to save setting '{key}': {exc}") from exc

    # ------------------------------------------------------------------
    # Daily tips
    # ------------------------------------------------------------------

    def get_tip_for_date(self, tip_date: date) -> Optional[str]:
        """Return the tip already recorded for a given calendar date, if any."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT tip_text FROM daily_tips WHERE tip_date = ?", (tip_date.isoformat(),)
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to read daily tip: {exc}") from exc
        return row["tip_text"] if row else None

    def save_tip_for_date(self, tip_date: date, tip_text: str) -> None:
        """Record which tip was shown on a given calendar date (idempotent)."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO daily_tips (tip_date, tip_text) VALUES (?, ?) "
                    "ON CONFLICT(tip_date) DO NOTHING",
                    (tip_date.isoformat(), tip_text),
                )
        except sqlite3.Error as exc:
            logger.error(f"Failed to save daily tip: {exc}")
            raise DatabaseError(f"Failed to save daily tip: {exc}") from exc
