"""
GraphRAG — Mode B from the spec.
Relationship-aware retrieval using the knowledge graph built in Phase 3.
Good for multi-hop questions: "how did X affect Y" style reasoning.
"""
import json
import logging

import networkx as nx
from openai import OpenAI

import config

logger = logging.getLogger("bizrag.graph_rag")

_llm_client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)

# How many "hops" (connection-jumps) to walk out from each found entity.
# 2 hops = entity -> its direct connections -> their connections too.
HOP_DEPTH = 2


class GraphRetrievalError(Exception):
    """Raised when a document's graph can't be found or loaded."""
    pass


def _load_graph(doc_id: str) -> nx.DiGraph:
    graph_path = config.GRAPH_DIR / f"{doc_id}.graphml"
    if not graph_path.exists():
        raise GraphRetrievalError(
            f"No knowledge graph found for doc_id={doc_id}. Has it finished ingesting?"
        )
    return nx.read_graphml(str(graph_path))


def _extract_query_entities(query: str) -> list[str]:
    """Ask the LLM which real-world entities the question is actually about."""
    prompt = f"""Extract the key entities (companies, people, financial
terms, concepts) mentioned or implied in this question.
Return ONLY JSON: {{"entities": ["entity1", "entity2"]}}

Question: {query}"""

    response = _llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("entities", [])


def _find_matching_nodes(entities: list[str], graph: nx.DiGraph) -> set:
    """
    Graph node names are exact text extracted by an LLM, so they rarely
    match a user's phrasing exactly. Instead of requiring an exact match,
    we do a loose substring match in both directions - catches cases like
    entity="revenue" matching a node called "Q3 2023 revenue".
    """
    matched = set()
    for entity in entities:
        entity_lower = entity.lower()
        for node in graph.nodes:
            node_lower = str(node).lower()
            if entity_lower in node_lower or node_lower in entity_lower:
                matched.add(node)
    return matched


def _subgraph_to_text(subgraph: nx.DiGraph) -> str:
    """Turn graph edges back into readable sentences for the LLM."""
    lines = []
    for source, target, data in subgraph.edges(data=True):
        relation = data.get("relation", "related to")
        lines.append(f"{source} --[{relation}]--> {target}")
    return "\n".join(lines)


def _rank_facts_by_relevance(query: str, facts: list, top_n: int = 25) -> list:
    """
    Hub nodes can return hundreds of facts. Instead of dumping all of them
    into the LLM prompt (burns tokens, drowns the signal), embed each fact
    and the query with the same local model used for vector RAG, then keep
    only the most relevant top_n - a lightweight reranking step.
    """
    if len(facts) <= top_n:
        return facts

    from ingestion import _embedder  # local import avoids a circular import at module load time

    query_vec = _embedder.embed_query(query)
    fact_vecs = _embedder.embed_documents(facts)

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    scored = [(fact, cosine_sim(query_vec, vec)) for fact, vec in zip(facts, fact_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [fact for fact, _ in scored[:top_n]]

def graph_rag_query(query: str, doc_id: str) -> dict:
    graph = _load_graph(doc_id)

    entities = _extract_query_entities(query)
    matched_nodes = _find_matching_nodes(entities, graph)

    if not matched_nodes:
        return {
            "answer": "I couldn't find any relevant entities in the knowledge graph for this question.",
            "graph_facts": [],
            "mode": "graph_rag",
        }

    # Walk HOP_DEPTH steps outward from every matched node, gathering
    # everything connected within that distance (multi-hop neighborhood).
    subgraph_nodes = set()
    for node in matched_nodes:
        # NetworkX's shortest_path only follows edge direction; treat the
        # graph as undirected here since a fact might be relevant whether
        # we found the "cause" node or the "effect" node.
        neighbours = nx.single_source_shortest_path_length(
            graph.to_undirected(), node, cutoff=HOP_DEPTH
        )
        subgraph_nodes.update(neighbours.keys())

    
    subgraph = graph.subgraph(subgraph_nodes)
    all_facts = _subgraph_to_text(subgraph).split("\n")
    ranked_facts = _rank_facts_by_relevance(query, all_facts)
    graph_context = "\n".join(ranked_facts)

    if not graph_context:
        return {
            "answer": "Found matching entities but no connected relationships in the graph.",
            "graph_facts": [],
            "mode": "graph_rag",
        }

    prompt = f"""Answer the question using ONLY the relationship facts below.
Each line shows a connection between two things. Trace connections across
multiple lines if needed to answer multi-part questions. If the answer
truly isn't supported by these facts, say "Not found in document."

Relationship facts:
{graph_context}

Question: {query}

Answer:"""

    response = _llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "graph_facts": ranked_facts,
        "mode": "graph_rag",
    }


