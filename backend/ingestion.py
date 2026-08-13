"""
Document ingestion pipeline.

Pipeline stages (mirrors the progress indicator the frontend will show):
  1. Extracting text      (PyMuPDF)
  2. Chunking             (RecursiveCharacterTextSplitter)
  3. Embedding + indexing (local sentence-transformers -> FAISS)
  4. Building knowledge graph  (stubbed here, built in Phase 3)
  5. Ready

Each stage updates metadata_store so the frontend can poll /documents/{id}
and render a live progress bar instead of a spinner.
"""
import logging
import uuid
from pathlib import Path
from knowledge_graph import build_knowledge_graph

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import config
import metadata_store

logger = logging.getLogger("bizrag.ingestion")

# Loaded once at import time, reused for every document — loading the
# model from disk on every request would be slow and wasteful.
_embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


class IngestionError(Exception):
    """Raised when any ingestion stage fails; carries the stage it failed at."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


def _set_status(doc_id: str, stage: str, **extra):
    record = metadata_store.get_document(doc_id) or {"doc_id": doc_id}
    record["status"] = stage
    record.update(extra)
    metadata_store.save_document(doc_id, record)


def extract_text(pdf_path: Path) -> str:
    """Stage 1: pull raw text out of every page of the PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise IngestionError("extracting_text", f"Could not open PDF: {e}")

    if doc.page_count == 0:
        raise IngestionError("extracting_text", "PDF has zero pages")

    text_parts = [page.get_text() for page in doc]
    doc.close()
    text = "\n".join(text_parts).strip()

    if not text:
        raise IngestionError(
            "extracting_text",
            "No extractable text found. This PDF may be scanned images — "
            "OCR is not yet implemented in this pipeline.",
        )
    return text


def chunk_text(text: str, doc_id: str, source: str) -> list:
    """Stage 2: split into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        [text], metadatas=[{"doc_id": doc_id, "source": source}]
    )
    if not chunks:
        raise IngestionError("chunking", "Splitter produced zero chunks")
    return chunks


def embed_and_index(chunks: list, doc_id: str) -> Path:
    """Stage 3: embed chunks locally and persist a FAISS index."""
    try:
        vectorstore = FAISS.from_documents(chunks, _embedder)
    except Exception as e:
        raise IngestionError("embedding", f"Embedding/index build failed: {e}")

    index_path = config.INDEX_DIR / f"{doc_id}_faiss"
    vectorstore.save_local(str(index_path))
    return index_path


def ingest_document(pdf_path: str, original_filename: str) -> dict:
    """
    Full pipeline entry point. Returns a summary dict.
    Called from a FastAPI BackgroundTask so /upload returns immediately
    and the client polls status via GET /documents/{doc_id}.
    """
    doc_id = str(uuid.uuid4())
    pdf_path = Path(pdf_path)

    metadata_store.save_document(doc_id, {
        "doc_id": doc_id,
        "filename": original_filename,
        "status": "queued",
    })

    try:
        _set_status(doc_id, "extracting_text")
        text = extract_text(pdf_path)

        _set_status(doc_id, "chunking")
        chunks = chunk_text(text, doc_id, source=original_filename)

        _set_status(doc_id, "embedding")
        index_path = embed_and_index(chunks, doc_id)

    
        _set_status(doc_id, "building_graph")
        graph = build_knowledge_graph(text, doc_id)
        entity_count = graph.number_of_nodes()
        relation_count = graph.number_of_edges()

        _set_status(
            doc_id,
            "ready",
            chunk_count=len(chunks),
            entity_count=entity_count,
            relation_count=relation_count,
            index_path=str(index_path),
            char_count=len(text),
        )
        logger.info("Ingested %s: %d chunks", doc_id, len(chunks))

    except IngestionError as e:
        _set_status(doc_id, "failed", error=e.message, failed_stage=e.stage)
        raise
    except Exception as e:
        _set_status(doc_id, "failed", error=str(e), failed_stage="unknown")
        raise

    return metadata_store.get_document(doc_id)