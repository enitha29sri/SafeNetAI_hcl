"""
rag.py

Complete Retrieval-Augmented Generation (RAG) pipeline for SafeNet AI's
cybersecurity knowledge base.

Pipeline stages:
    1. Load  - PyMuPDFLoader reads every PDF in data/pdfs/
    2. Split - RecursiveCharacterTextSplitter breaks pages into overlapping chunks
    3. Embed - SentenceTransformerEmbeddings (all-MiniLM-L6-v2) vectorizes chunks
    4. Index - FAISS stores vectors for fast similarity search
    5. Retrieve - top-k most relevant chunks are returned for a given query

The FAISS index is persisted to disk (data/vector_store/) so it only needs
to be rebuilt when the source PDFs change, not on every app restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from models.embedding_model import EmbeddingModelError, SentenceTransformerEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)

# Filename used for the FAISS index saved under settings.vector_store_dir.
_INDEX_NAME = "safenet_faiss_index"


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline fails to build, load, or query the index."""


class RAGPipeline:
    """
    End-to-end RAG pipeline: PDF ingestion, chunking, embedding, FAISS
    indexing, and similarity-based retrieval for SafeNet AI's cybersecurity
    knowledge base.
    """

    def __init__(self, embeddings: Optional[SentenceTransformerEmbeddings] = None) -> None:
        """
        Args:
            embeddings: An existing embeddings instance to reuse, or None to
                create a fresh SentenceTransformerEmbeddings.

        Raises:
            RAGPipelineError: If the embedding model fails to load.
        """
        try:
            self._embeddings = embeddings or SentenceTransformerEmbeddings()
        except EmbeddingModelError as exc:
            raise RAGPipelineError(str(exc)) from exc

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._vector_store: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Loading & chunking
    # ------------------------------------------------------------------

    def load_pdf_documents(self, pdf_dir: Optional[Path] = None) -> list[Document]:
        """
        Load every PDF in the given directory into LangChain Document objects.

        Args:
            pdf_dir: Directory to scan for `.pdf` files. Defaults to
                settings.pdf_source_dir.

        Returns:
            A flat list of per-page Document objects across all PDFs found.
            Returns an empty list (with a warning logged) if no PDFs exist.
        """
        directory = pdf_dir or settings.pdf_source_dir
        pdf_paths = sorted(directory.glob("*.pdf"))

        if not pdf_paths:
            logger.warning(f"No PDF files found in '{directory}'. Knowledge base is empty.")
            return []

        documents: list[Document] = []
        for pdf_path in pdf_paths:
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                loaded = loader.load()
                documents.extend(loaded)
                logger.info(f"Loaded {len(loaded)} page(s) from '{pdf_path.name}'.")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to load PDF '{pdf_path.name}': {exc}")
                # Skip the bad file rather than aborting the whole ingestion run.
                continue

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split loaded documents into overlapping chunks for embedding.

        Args:
            documents: Raw, per-page Document objects from load_pdf_documents().

        Returns:
            A list of smaller Document chunks, each carrying the original
            document's metadata (source file, page number).
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document page(s) into {len(chunks)} chunk(s).")
        return chunks

    # ------------------------------------------------------------------
    # Index building & persistence
    # ------------------------------------------------------------------

    def build_index(self, pdf_dir: Optional[Path] = None) -> int:
        """
        Build a fresh FAISS index from all PDFs in the source directory and
        persist it to disk.

        Args:
            pdf_dir: Directory to scan for `.pdf` files. Defaults to
                settings.pdf_source_dir.

        Returns:
            The number of chunks indexed. Returns 0 if no PDFs were found
            (no index is created in that case).

        Raises:
            RAGPipelineError: If embedding or index construction fails.
        """
        documents = self.load_pdf_documents(pdf_dir)
        chunks = self.split_documents(documents)

        if not chunks:
            logger.warning("No chunks to index. Skipping FAISS index build.")
            return 0

        try:
            self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to build FAISS index: {exc}")
            raise RAGPipelineError(f"Failed to build FAISS index: {exc}") from exc

        self.save_index()
        logger.info(f"FAISS index built and saved with {len(chunks)} chunk(s).")
        return len(chunks)

    def save_index(self) -> None:
        """
        Persist the current in-memory FAISS index to settings.vector_store_dir.

        Raises:
            RAGPipelineError: If no index has been built yet, or saving fails.
        """
        if self._vector_store is None:
            raise RAGPipelineError("No index to save. Call build_index() first.")

        try:
            settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(settings.vector_store_dir), index_name=_INDEX_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to save FAISS index: {exc}")
            raise RAGPipelineError(f"Failed to save FAISS index: {exc}") from exc

    def load_index(self) -> bool:
        """
        Load a previously persisted FAISS index from disk, if one exists.

        Returns:
            True if an index was found and loaded, False if no persisted
            index exists yet (caller should then run build_index()).

        Raises:
            RAGPipelineError: If an index file exists but fails to load
                (e.g. corrupted or built with a different embedding model).
        """
        index_file = settings.vector_store_dir / f"{_INDEX_NAME}.faiss"
        if not index_file.exists():
            logger.info("No persisted FAISS index found on disk.")
            return False

        try:
            self._vector_store = FAISS.load_local(
                str(settings.vector_store_dir),
                self._embeddings,
                index_name=_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load FAISS index: {exc}")
            raise RAGPipelineError(f"Failed to load FAISS index: {exc}") from exc

        logger.info("FAISS index loaded from disk.")
        return True

    def ensure_index_ready(self) -> None:
        """
        Guarantee an index is loaded in memory, building one from scratch
        if no persisted index exists yet. Call this once at app startup.

        Raises:
            RAGPipelineError: If building a fresh index also fails.
        """
        if self.load_index():
            return
        self.build_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """
        Retrieve the most relevant chunks of text for a given query.

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to retrieve. Defaults to
                settings.retriever_top_k.

        Returns:
            A list of raw chunk text strings, most relevant first. Returns
            an empty list if no index is available or the query is empty.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        if self._vector_store is None:
            logger.warning("retrieve() called before an index was built/loaded.")
            return []

        k = top_k or settings.retriever_top_k

        try:
            results = self._vector_store.similarity_search(cleaned_query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Similarity search failed: {exc}")
            raise RAGPipelineError(f"Failed to search the knowledge base: {exc}") from exc

        return [doc.page_content for doc in results]

    def retrieve_with_sources(
        self, query: str, top_k: Optional[int] = None
    ) -> list[dict[str, str]]:
        """
        Retrieve relevant chunks along with their originating file/page,
        useful for showing citations in the UI.

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to retrieve. Defaults to
                settings.retriever_top_k.

        Returns:
            A list of dicts: {"content": str, "source": str, "page": str}.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query or self._vector_store is None:
            return []

        k = top_k or settings.retriever_top_k

        try:
            results = self._vector_store.similarity_search(cleaned_query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Similarity search (with sources) failed: {exc}")
            raise RAGPipelineError(f"Failed to search the knowledge base: {exc}") from exc

        return [
            {
                "content": doc.page_content,
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "page": str(doc.metadata.get("page", "N/A")),
            }
            for doc in results
        ]

    @property
    def is_ready(self) -> bool:
        """Whether a FAISS index is currently loaded in memory."""
        return self._vector_store is not None
