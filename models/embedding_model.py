"""
models/embedding_model.py

Wraps a local `sentence-transformers` model (all-MiniLM-L6-v2 by default)
behind LangChain's `Embeddings` interface, so it can be passed directly to
a LangChain FAISS vector store without any adapter code elsewhere.

Using a local embedding model (instead of an API-based one) keeps the RAG
pipeline free, fast, and fully offline-capable once the model weights are
cached, which matters for a college demo where network access may be
unreliable.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load or embed text."""


class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings backed by a local SentenceTransformer model.

    Implements the two methods LangChain's `Embeddings` ABC requires:
    `embed_documents` (batch) and `embed_query` (single string).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Load the SentenceTransformer model into memory.

        Args:
            model_name: HuggingFace model identifier. Defaults to the
                value configured in settings (all-MiniLM-L6-v2).

        Raises:
            EmbeddingModelError: If the model fails to download or load.
        """
        self._model_name = model_name or settings.embedding_model_name

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load embedding model '{self._model_name}': {exc}")
            raise EmbeddingModelError(
                f"Could not load embedding model '{self._model_name}': {exc}"
            ) from exc

        logger.info(f"Embedding model '{self._model_name}' loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents/chunks for indexing.

        Args:
            texts: List of raw text chunks to embed.

        Returns:
            A list of embedding vectors (one per input text), as plain
            Python lists of floats (required by LangChain's FAISS wrapper).

        Raises:
            EmbeddingModelError: If embedding fails.
        """
        if not texts:
            return []

        try:
            vectors = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to embed {len(texts)} document(s): {exc}")
            raise EmbeddingModelError(f"Failed to embed documents: {exc}") from exc

        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for similarity search.

        Args:
            text: The user's search query.

        Returns:
            The embedding vector as a plain Python list of floats.

        Raises:
            EmbeddingModelError: If embedding fails.
        """
        if not text or not text.strip():
            raise EmbeddingModelError("Cannot embed an empty query string.")

        try:
            vector = self._model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to embed query: {exc}")
            raise EmbeddingModelError(f"Failed to embed query: {exc}") from exc

        return vector.tolist()

    @property
    def model_name(self) -> str:
        """The HuggingFace identifier of the loaded model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """The output embedding vector size for the loaded model."""
        return self._model.get_sentence_embedding_dimension()
