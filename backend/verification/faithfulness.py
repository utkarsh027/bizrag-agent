"""
Faithfulness verifier — Phase 5, the core research contribution.

Checks whether a generated answer is actually supported by the retrieved
evidence, at the individual claim level (not just a single answer-level
score). Uses a local NLI (Natural Language Inference) model — free, runs
on CPU, no API cost, and independent of the LLM that generated the answer.
"""
import json
import logging

from transformers import pipeline
from openai import OpenAI

import config

logger = logging.getLogger("bizrag.faithfulness")

_llm_client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)

# Loaded once at import time — loading a model from disk on every request
# would be slow. Runs on CPU (device=-1); set device=0 if you ever have a GPU.
_nli_pipeline = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base",
    device=-1,
)


def extract_claims(answer: str) -> list:
    """Split a generated answer into individual, independently-checkable claims."""
    prompt = f"""Split this answer into individual factual claims, each a
complete standalone sentence. Return ONLY JSON: {{"claims": ["claim1", "claim2"]}}
If the answer is a single simple statement, return it as one claim.

Answer: {answer}"""

    response = _llm_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("claims", [answer])  # fall back to the whole answer as one claim


def _check_claim(claim: str, context: str) -> dict:
    """
    Run one claim through the local NLI model against the evidence.
    NLI models expect a specific format: premise (the evidence) and
    hypothesis (the claim), usually joined with [SEP].
    """
    nli_input = f"{context[:1500]} [SEP] {claim}"

    result = _nli_pipeline(nli_input, truncation=True, max_length=512)[0]
    label = result["label"].upper()
    score = result["score"]

    if label == "ENTAILMENT" and score > 0.7:
        verdict = "SUPPORTED"
    elif label == "CONTRADICTION":
        verdict = "NOT_SUPPORTED"
    else:
        verdict = "UNCERTAIN"

    return {"claim": claim, "verdict": verdict, "confidence": round(score, 3), "raw_label": label}


def verify_faithfulness(answer: str, retrieved_evidence: list) -> dict:
    """
    Main entry point. Takes a generated answer and the evidence it was
    supposedly based on (chunks from vector RAG, or facts from GraphRAG),
    and returns a claim-by-claim faithfulness breakdown plus an overall score.
    """
    claims = extract_claims(answer)
    context = " ".join(retrieved_evidence)

    if not context.strip():
        return {
            "faithfulness_score": 0.0,
            "verdict": "LOW",
            "claims": [],
            "note": "No evidence provided to verify against.",
        }

    results = [_check_claim(claim, context) for claim in claims]

    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    faithfulness_score = supported / len(results) if results else 0.0

    overall_verdict = (
        "HIGH" if faithfulness_score > 0.8
        else "MEDIUM" if faithfulness_score > 0.5
        else "LOW"
    )

    return {
        "faithfulness_score": round(faithfulness_score, 3),
        "verdict": overall_verdict,
        "claims": results,
    }