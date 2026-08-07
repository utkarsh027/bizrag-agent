"""
Minimal metadata persistence layer.

Design note: we use a tiny JSON store behind a clean interface instead of
jumping straight to a database. We don't yet know our full schema needs —
faithfulness history, graph stats, etc. all get added in later phases.
Swapping this JSON backend for Postgres/Supabase later only touches this
one file, not main.py or ingestion.py, because callers only ever import
these four functions.
"""
import json
import threading
from datetime import datetime, timezone
from typing import Optional

from config import METADATA_FILE

_lock = threading.Lock()


def _read() -> dict:
    if not METADATA_FILE.exists():
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def _write(data: dict) -> None:
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_document(doc_id: str, record: dict) -> None:
    with _lock:
        data = _read()
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        data[doc_id] = record
        _write(data)


def get_document(doc_id: str) -> Optional[dict]:
    with _lock:
        return _read().get(doc_id)


def list_documents() -> list[dict]:
    with _lock:
        data = _read()
        return [{"doc_id": k, **v} for k, v in data.items()]


def delete_document(doc_id: str) -> bool:
    with _lock:
        data = _read()
        if doc_id in data:
            del data[doc_id]
            _write(data)
            return True
        return False