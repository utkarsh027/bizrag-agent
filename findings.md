# BizRAG-Agent — Research Findings Log

Raw observations from building and testing this system, in chronological order.
This is the source material for the research proposal (Part D) — write the
proposal itself in your own words, but pull real examples from here.

---

## Finding 1 — Retrieval succeeds, generation still fails
**Date:** Aug 10, 2026
**Phase:** Phase 2 (Vector RAG query endpoint)

**What I did:**
Asked the system: *"What is a transformer model?"* against a 26,601-character
PDF (Transformers_Complete_Guide.pdf), using top_k=5 retrieval.

**What happened:**
The retriever correctly pulled the chunk containing the answer — page 4 of the
document literally states "The original Transformer is an encoder-decoder
model designed for sequence-to-sequence tasks..." — this chunk was in the
returned `sources` array. But the LLM's generated answer said:

> "The context does not provide a direct definition... Therefore, the answer
> is: Not found in document."

**Why this matters (my interpretation — replace/expand with your own):**
Retrieval and generation are two separate points of failure, and they can
fail independently. My retrieval component worked correctly — the relevant
evidence was found. But the generation step misread its own retrieved
context, likely because the question's phrasing ("what is X") didn't
lexically match the document's phrasing ("X is described as..."). This is
a different problem from the "multi-hop reasoning" failure GraphRAG is
built to solve — GraphRAG wouldn't fix this at all, since it's not a
retrieval problem here, it's a generation-faithfulness problem.

**Connection to the verifier (Phase 5):**
This is exactly the kind of error a claim-level NLI faithfulness check
should catch: the retrieved context entails a definition, but the
generated answer contradicts it by claiming "not found." An answer-level
faithfulness score might have missed this (the model *did* generate mostly
accurate reasoning about the architecture before contradicting itself at
the end) — a claim-level check would isolate the "not found" claim
specifically and flag it as NOT_SUPPORTED / contradicted by evidence.

**Proposal section this feeds:** Section 1 (Problem), Section 3 (Proposal), Section 4 (Preliminary Evidence)

---

## Finding 2 — [next observation goes here]
**Date:**
**Phase:**

**What I did:**


**What happened:**


**Why this matters:**


**Proposal section this feeds:**

---
