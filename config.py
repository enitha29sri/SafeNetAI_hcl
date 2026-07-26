"""
config.py

Centralized application configuration for SafeNet AI.

This module loads environment variables (via python-dotenv) once and exposes
a single, immutable, type-hinted `Settings` object that every other module
in the project imports. This avoids scattering `os.getenv()` calls across
the codebase and gives us one source of truth for paths, model names,
feature flags, and secrets.

Usage:
    from config import settings

    api_key = settings.gemini_api_key
    db_path = settings.database_path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment variables from .env (if present) into os.environ.
# This must run before we read any variables below.
# ------------------------------------------------------------------
load_dotenv()

# Root directory of the project (the folder this file lives in).
BASE_DIR: Path = Path(__file__).resolve().parent


def _get_bool(key: str, default: bool = False) -> bool:
    """
    Safely parse a boolean-like environment variable.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is unset or invalid.

    Returns:
        The parsed boolean value.
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    """
    Safely parse an integer environment variable.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is unset or invalid.

    Returns:
        The parsed integer value.
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings container.

    All configuration for SafeNet AI is centralized here so that other
    modules (chatbot.py, rag.py, services/*, utils/*) have a single,
    predictable import path for configuration values instead of calling
    os.getenv() directly throughout the codebase.

    Attributes:
        app_name: Human-readable application name shown in the UI.
        app_version: Semantic version string of the application.
        app_env: Deployment environment identifier (development/production).
        app_debug: Whether verbose debug behavior is enabled.
        gemini_api_key: Secret API key for Google Gemini.
        gemini_model_name: Which Gemini model to call for generation.
        embedding_model_name: SentenceTransformer model used for embeddings.
        base_dir: Absolute path to the project root.
        pdf_source_dir: Directory containing source PDFs for the RAG KB.
        vector_store_dir: Directory where the FAISS index is persisted.
        chunk_size: Character length of each RAG text chunk.
        chunk_overlap: Overlap (in characters) between consecutive chunks.
        retriever_top_k: Number of chunks retrieved per RAG query.
        database_path: Filesystem path to the SQLite database file.
        log_level: Logging verbosity level (e.g. INFO, DEBUG).
        log_file_path: Filesystem path where log files are written.
        max_chat_input_length: Hard cap on characters accepted from chat input.
        session_timeout_minutes: Idle session timeout, in minutes.
    """

    # --- Application metadata ---
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "SafeNet AI"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_debug: bool = field(default_factory=lambda: _get_bool("APP_DEBUG", True))

    # --- Gemini LLM ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    )

    # --- Embeddings ---
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )

    # --- Paths ---
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    pdf_source_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("PDF_SOURCE_DIR", "data/pdfs")
    )
    vector_store_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    )
    database_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("DATABASE_PATH", "database/sqlite.db")
    )
    log_file_path: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("LOG_FILE_PATH", "logs/safenet_ai.log")
    )

    # --- RAG pipeline tuning ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))
    retriever_top_k: int = field(default_factory=lambda: _get_int("RETRIEVER_TOP_K", 4))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- Security / limits ---
    max_chat_input_length: int = field(
        default_factory=lambda: _get_int("MAX_CHAT_INPUT_LENGTH", 2000)
    )
    session_timeout_minutes: int = field(
        default_factory=lambda: _get_int("SESSION_TIMEOUT_MINUTES", 30)
    )

    def validate(self) -> list[str]:
        """
        Validate required configuration values.

        Returns:
            A list of human-readable error messages. An empty list means
            the configuration is valid enough to run the application.
        """
        errors: list[str] = []

        if not self.gemini_api_key:
            errors.append(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(see .env.example for the required format)."
            )

        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )

        if self.retriever_top_k < 1:
            errors.append("RETRIEVER_TOP_K must be at least 1.")

        return errors

    def ensure_directories(self) -> None:
        """
        Create all directories required by the application if they do
        not already exist. Safe to call multiple times.
        """
        directories = [
            self.pdf_source_dir,
            self.vector_store_dir,
            self.database_path.parent,
            self.log_file_path.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Singleton settings instance imported throughout the application.
# ------------------------------------------------------------------
settings = Settings()
settings.ensure_directories()


if __name__ == "__main__":
    # Simple manual sanity check: `python config.py`
    issues = settings.validate()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"{settings.app_name} v{settings.app_version} configuration OK.")
        print(f"Base directory: {settings.base_dir}")
