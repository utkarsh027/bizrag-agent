"""
Phase 6 — Agentic orchestrator. LLM decides which retrieval strategy
(vector RAG, GraphRAG, or both) fits a given question, executes it,
and returns the answer with its reasoning.
"""
import json
import logging

from openai import OpenAI


import config
from retrieval.vector_rag import vector_rag_query, RetrievalError
from retrieval.graph_rag import graph_rag_query, GraphRetrievalError

logger = logging.getLogger("bizrag.agent")

_llm_client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)

SYSTEM_PROMPT = """You are a business intelligence routing agent. Given a
question about a document, decide which retrieval strategy fits best:

- "vector_rag": direct facts, definitions, specific numbers, single-topic questions
- "graph_rag": relationships, multi-hop reasoning, cause-and-effect, "how did X affect Y"
- "both": complex questions that may need both direct facts AND relationships

Respond ONLY with JSON: {"strategy": "vector_rag" | "graph_rag" | "both", "reasoning": "one sentence why"}"""


def agentic_query(query: str, doc_id: str) -> dict:
    routing_response = _llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    decision = json.loads(routing_response.choices[0].message.content)
    strategy = decision.get("strategy", "vector_rag")
    reasoning = decision.get("reasoning", "")

    try:
        if strategy == "vector_rag":
            result = vector_rag_query(query, doc_id)
        elif strategy == "graph_rag":
            result = graph_rag_query(query, doc_id)
        elif strategy == "both":
            vec_result = vector_rag_query(query, doc_id)
            graph_result = graph_rag_query(query, doc_id)
            # Simple synthesis: let the LLM combine both answers into one.
            synth_prompt = f"""Combine these two answers to the same question
into one clear, non-redundant answer. If they conflict, note the
discrepancy rather than picking one arbitrarily.

Question: {query}

Vector search answer: {vec_result['answer']}

Graph search answer: {graph_result['answer']}

Combined answer:"""
            synth_response = _llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": synth_prompt}],
                temperature=0.1,
            )
            result = {
                "answer": synth_response.choices[0].message.content,
                "sources": vec_result.get("sources", []) + graph_result.get("graph_facts", []),
                "mode": "both",
            }
        else:
            result = vector_rag_query(query, doc_id)  # safe fallback
    except (RetrievalError, GraphRetrievalError) as e:
        raise

    result["agent_strategy"] = strategy
    result["agent_reasoning"] = reasoning
    return result

