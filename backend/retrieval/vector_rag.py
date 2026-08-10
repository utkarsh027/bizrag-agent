"""
Vector RAG — Mode A from the spec.
Fast, semantic-similarity retrieval. Good for direct factual questions.
"""
import logging

from langchain_community.vectorstores import FAISS
from openai import OpenAI

import config
from ingestion import _embedder  # reuse the same model used at indexing time

logger = logging.getLogger("bizrag.vector_rag")

# Groq client — OpenAI-compatible SDK pointed at Groq's endpoint instead of OpenAI's
_llm_client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)


class RetrievalError(Exception):
    """Raised when a document's index can't be found or loaded."""
    pass


def _load_index(doc_id: str):
    index_path = config.INDEX_DIR / f"{doc_id}_faiss"
    if not index_path.exists():
        raise RetrievalError(
            f"No index found for doc_id={doc_id}. Has it finished ingesting?"
        )
    # allow_dangerous_deserialization=True is required by recent FAISS/LangChain
    # versions. It's safe here because WE are the ones who created this index
    # file in the first place (in ingestion.py) — we're not loading an
    # untrusted file from someone else.
    return FAISS.load_local(
        str(index_path), _embedder, allow_dangerous_deserialization=True
    )


def vector_rag_query(query: str, doc_id: str, top_k: int = 5) -> dict:
    vectorstore = _load_index(doc_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "I couldn't find any relevant content in this document.",
            "sources": [],
            "mode": "vector_rag",
        }

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""Answer the question based ONLY on the context below.
If the answer is not in the context, say "Not found in document." Do not make anything up.

Context:
{context}

Question: {query}

Answer:"""

    response = _llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [d.page_content[:200] for d in docs],
        "mode": "vector_rag",
    }