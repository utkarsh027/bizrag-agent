"""
Centralized configuration for BizRAG-Agent backend.
Loads from environment variables (.env) so no secrets live in code.

Free-tier stack:
- Embeddings: sentence-transformers (local, free, no API key)
- LLM: Groq API (free tier, OpenAI-compatible, very fast)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Storage paths ---
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "indexes"
GRAPH_DIR = BASE_DIR / "graphs"
METADATA_FILE = BASE_DIR / "metadata.json"

for d in (UPLOAD_DIR, INDEX_DIR, GRAPH_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Ingestion params ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Local, free embedding model (runs on CPU, downloads once ~90MB)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- LLM (Groq — free tier, OpenAI-compatible API) ---
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# --- Server ---
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

if not GROQ_API_KEY:
    import warnings
    warnings.warn(
        "GROQ_API_KEY is not set. LLM calls (query, agent, entity extraction) "
        "will fail until you add it to backend/.env. Embeddings will still "
        "work fine since they run locally."
    )