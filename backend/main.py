"""
BizRAG-Agent — FastAPI backend entrypoint.

Phase 1 scope: document upload + ingestion status.
Routes added in later phases (query, verify, graph, compare) get
registered here as they're built.
"""
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional
from retrieval.vector_rag import vector_rag_query, RetrievalError
from retrieval.graph_rag import graph_rag_query, GraphRetrievalError
from retrieval.vector_rag import RetrievalError
from retrieval.graph_rag import GraphRetrievalError
from verification.faithfulness import verify_faithfulness
from agent import agentic_query




from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import ingestion
import metadata_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bizrag.main")

app = FastAPI(
    title="BizRAG-Agent API",
    description="Agentic Vector RAG + GraphRAG business intelligence backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 50


class DocumentStatus(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    entity_count: Optional[int] = None
    char_count: Optional[int] = None
    error: Optional[str] = None
    failed_stage: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "bizrag-agent-backend"}


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are supported.")

    tmp_name = f"{uuid.uuid4()}.pdf"
    tmp_path = config.UPLOAD_DIR / tmp_name
    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    with open(tmp_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit."
                )
            out.write(chunk)

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, detail="Uploaded file is empty.")

    background_tasks.add_task(
        _run_ingestion_safely, str(tmp_path), file.filename
    )

    return {
        "message": "Upload received, processing started.",
        "filename": file.filename,
        "poll_hint": "GET /documents to find your doc_id once processing starts.",
    }

class QueryRequest(BaseModel):
    query: str
    doc_id: str
    top_k: int = 5


@app.post("/query")
def query_document(request: QueryRequest):
    record = metadata_store.get_document(request.doc_id)
    if not record:
        raise HTTPException(404, detail="Document not found.")
    if record.get("status") != "ready":
        raise HTTPException(
            409,
            detail=f"Document is not ready yet (status: {record.get('status')}).",
        )

    try:
        result = vector_rag_query(request.query, request.doc_id, request.top_k)
    except RetrievalError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        logger.exception("Query failed for doc_id=%s", request.doc_id)
        raise HTTPException(500, detail=f"Query failed: {e}")

    return result

@app.post("/graph-query")
def graph_query_document(request: QueryRequest):
    record = metadata_store.get_document(request.doc_id)
    if not record:
        raise HTTPException(404, detail="Document not found.")
    if record.get("status") != "ready":
        raise HTTPException(
            409,
            detail=f"Document is not ready yet (status: {record.get('status')}).",
        )

    try:
        result = graph_rag_query(request.query, request.doc_id)
    except GraphRetrievalError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        logger.exception("Graph query failed for doc_id=%s", request.doc_id)
        raise HTTPException(500, detail=f"Graph query failed: {e}")

    return result

@app.post("/agent-query")
def agent_query_document(request: QueryRequest):
    record = metadata_store.get_document(request.doc_id)
    if not record:
        raise HTTPException(404, detail="Document not found.")
    if record.get("status") != "ready":
        raise HTTPException(409, detail=f"Document is not ready yet (status: {record.get('status')}).")

    try:
        result = agentic_query(request.query, request.doc_id)
    except (RetrievalError, GraphRetrievalError) as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        logger.exception("Agent query failed for doc_id=%s", request.doc_id)
        raise HTTPException(500, detail=f"Agent query failed: {e}")

    return result

class VerifyRequest(BaseModel):
    answer: str
    evidence: list[str]


@app.post("/verify")
def verify_answer(request: VerifyRequest):
    try:
        result = verify_faithfulness(request.answer, request.evidence)
    except Exception as e:
        logger.exception("Faithfulness verification failed")
        raise HTTPException(500, detail=f"Verification failed: {e}")

    return result






def _run_ingestion_safely(tmp_path: str, filename: str):
    try:
        ingestion.ingest_document(tmp_path, filename)
    except ingestion.IngestionError as e:
        logger.warning("Ingestion failed for %s at stage %s: %s", filename, e.stage, e.message)
    except Exception:
        logger.exception("Unexpected ingestion failure for %s", filename)


@app.get("/documents", response_model=list[DocumentStatus])
def list_documents():
    return metadata_store.list_documents()


@app.get("/documents/{doc_id}", response_model=DocumentStatus)
def get_document(doc_id: str):
    record = metadata_store.get_document(doc_id)
    if not record:
        raise HTTPException(404, detail="Document not found.")
    return record


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    record = metadata_store.get_document(doc_id)
    if not record:
        raise HTTPException(404, detail="Document not found.")

    index_path = Path(record.get("index_path", ""))
    if index_path.exists():
        shutil.rmtree(index_path, ignore_errors=True)

    metadata_store.delete_document(doc_id)
    return {"message": "Deleted", "doc_id": doc_id}