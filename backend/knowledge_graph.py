"""
Knowledge graph construction — Phase 3.

Extracts (subject, relation, object) triples from document text using the
LLM, then builds a NetworkX directed graph. This graph powers GraphRAG
(Phase 4) — relationship-aware retrieval for multi-hop questions that
vector similarity search alone struggles with.
"""
import json
import logging

import networkx as nx
from openai import OpenAI

import config

logger = logging.getLogger("bizrag.knowledge_graph")

_llm_client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)

# How much text we hand the LLM per extraction call. Larger sections mean
# fewer API calls (faster, less rate-limit risk on the free tier), but the
# LLM may miss relationships buried deep in a big block of text.
SECTION_SIZE = 3000

EXTRACTION_PROMPT = """Extract entity relationships from the text below.
Return ONLY a JSON object in this exact format, nothing else:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Focus on concrete, factual relationships (companies, people, financial
figures, causes and effects). Skip vague or trivial connections. Extract
at most 15 triples. If there are no clear relationships, return
{{"triples": []}}.

Text:
{text}"""


def _split_into_sections(text: str, size: int = SECTION_SIZE) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _extract_triples_from_section(section: str) -> list:
    try:
        response = _llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=section)}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        triples = data.get("triples", [])
        # Basic validation — skip malformed entries instead of crashing
        # the whole pipeline over one bad triple.
        return [
            t for t in triples
            if isinstance(t, dict) and t.get("subject") and t.get("relation") and t.get("object")
        ]
    except Exception as e:
        logger.warning("Triple extraction failed for a section: %s", e)
        return []


def build_knowledge_graph(text: str, doc_id: str) -> nx.DiGraph:
    """
    Splits the full document text into sections, extracts entity
    relationships from each via the LLM, and assembles them into one
    directed graph. Saves the graph to disk as GraphML for later reuse.
    """
    sections = _split_into_sections(text)
    G = nx.DiGraph()

    for i, section in enumerate(sections):
        triples = _extract_triples_from_section(section)
        logger.info("Section %d/%d: extracted %d triples", i + 1, len(sections), len(triples))
        for t in triples:
            G.add_edge(t["subject"], t["object"], relation=t["relation"], doc_id=doc_id)

    graph_path = config.GRAPH_DIR / f"{doc_id}.graphml"
    nx.write_graphml(G, str(graph_path))
    logger.info(
        "Saved graph for %s: %d nodes, %d edges",
        doc_id, G.number_of_nodes(), G.number_of_edges()
    )

    return G