"""
Phase 9 — Compare mode. Runs vector RAG, GraphRAG, and the agent on the
same question, verifies each answer's faithfulness, and returns all
three side by side. This is the primary evidence-generation endpoint
for evaluating retrieval strategies against each other.
"""
import logging

from retrieval.vector_rag import vector_rag_query, RetrievalError
from retrieval.graph_rag import graph_rag_query, GraphRetrievalError
from agent import agentic_query
from verification.faithfulness import verify_faithfulness

logger = logging.getLogger("bizrag.compare")


def _safe_run(fn, *args):
    """Run a retrieval function, catching errors so one mode failing
    doesn't take down the whole comparison."""
    try:
        return fn(*args)
    except (RetrievalError, GraphRetrievalError) as e:
        return {"answer": None, "error": str(e), "mode": "error"}
    except Exception as e:
        logger.exception("Compare mode: a retrieval function failed")
        return {"answer": None, "error": str(e), "mode": "error"}


def _verify_result(result: dict) -> dict:
    """Attach a faithfulness check to a retrieval result, using whichever
    evidence field it has (vector RAG uses 'sources', GraphRAG uses
    'graph_facts')."""
    if not result.get("answer"):
        result["faithfulness"] = None
        return result

    evidence = result.get("sources") or result.get("graph_facts") or []
    result["faithfulness"] = verify_faithfulness(result["answer"], evidence)
    return result


def compare_query(query: str, doc_id: str) -> dict:
    vector_result = _verify_result(_safe_run(vector_rag_query, query, doc_id))
    graph_result = _verify_result(_safe_run(graph_rag_query, query, doc_id))
    agent_result = _verify_result(_safe_run(agentic_query, query, doc_id))

    return {
        "query": query,
        "doc_id": doc_id,
        "vector_rag": vector_result,
        "graph_rag": graph_result,
        "agentic": agent_result,
    }