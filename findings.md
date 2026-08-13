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

## Finding 2 — Confirming the phrasing-sensitivity hypothesis
**Date:** Aug 10, 2026
**Phase:** Phase 2 (Vector RAG query endpoint)

**What I did:**
Re-asked essentially the same question from Finding 1, but rephrased to
match the document's own language: "Describe the transformer architecture"
instead of "What is a transformer model?" Same document, same top_k=5.

**What happened:**
This time the system gave a complete, accurate, well-structured answer —
correctly describing the encoder-decoder structure, the 6x stacking, the
sub-layers (self-attention, add & norm, feed-forward), and residual
connections, all clearly grounded in the retrieved source chunks.

**Why this matters (my interpretation — replace with your own):**
This confirms the hypothesis from Finding 1: the earlier failure wasn't a
retrieval problem (the right chunks were found both times), it was a
question-phrasing sensitivity problem in the generation step. The LLM
appears to weight lexical overlap between the question and context more
than semantic equivalence when deciding whether an answer "exists" in the
context. This is a specific, testable failure mode I can point to directly
rather than a vague claim.

**Proposal section this feeds:** Section 1 (Problem — strengthens the concrete example), Section 4 (Preliminary Evidence — this is a controlled before/after comparison, which is more convincing than a single anecdote)

----
## Finding 3 — Free-tier pipeline scales to a real 80-page SEC filing
**Date:** Aug 13, 2026
**Phase:** Phase 3 (Knowledge graph extraction)

**What I did:**
Uploaded Apple's real, official 2023 Form 10-K (80 pages, downloaded
directly from investor.apple.com) — a genuine SEC-filed annual report,
not a synthetic test document — and ran it through the full ingestion
pipeline including knowledge graph extraction.

**What happened:**
Successfully processed end-to-end on the free stack: 272,866 characters
extracted, split into 358 chunks for vector search, and 1,144 entities
extracted into the knowledge graph via ~91 separate Groq API calls
(one per ~3000-character section). No errors, no rate limiting, no
failures at any pipeline stage.

**Why this matters (my interpretation — replace with your own):**
This demonstrates the free-tier architecture isn't just a toy that works
on small test files — it scales to a real, dense, 80-page financial
document roughly 10x larger than my initial test PDF, with zero cost.
This is meaningful preliminary evidence that a free/local stack (Groq +
sentence-transformers) is a viable foundation for a business intelligence
RAG system, not just an academic proof-of-concept limited to toy inputs.

**Proposal section this feeds:** Section 4 (Preliminary Evidence — direct evidence of scalability on a real business document), Section 6 (motivates using real SEC filings like FinQA/FinanceBench in Year 1 evaluation)

--
## Finding 4 — GraphRAG retrieves the right facts but drowns them in noise (hub-node explosion)
**Date:** Aug 13, 2026
**Phase:** Phase 4 (GraphRAG query endpoint)

**What I did:**
Asked the same question to both systems on the real Apple 10-K:
"How do risk factors affect Apple's revenue?"
- Vector RAG: "Not found in document" (Finding 3)
- GraphRAG: also "Not found in document," despite retrieving 300+ graph facts

**What happened:**
Unlike vector RAG, GraphRAG's retrieval actually surfaced directly relevant
facts: "foreign exchange rate risk --[affects]--> net sales" and
"...--[affects]--> gross margins" were both present in the retrieved
subgraph. But the final answer still said "not found." Inspecting the
full `graph_facts` output revealed why: because "Apple Inc." is a hub
node connected to hundreds of other entities in the document (board
members, lease terms, tax details, product launches...), a 2-hop walk
from it pulled in over 300 relationship lines - the two genuinely
relevant facts were buried in a mostly-irrelevant mass of context.

**Why this matters (my interpretation — replace with your own):**
This is a different failure mode from Findings 1-2 (phrasing sensitivity)
- this is "hub-node explosion" in multi-hop graph traversal: naively
walking N hops from a highly-connected node doesn't give a focused
neighborhood, it gives a large fraction of the whole graph. This is a
known challenge in the GraphRAG literature - Edge et al. (2024) address
a related problem with community detection/summarization rather than
raw multi-hop traversal, which my simple implementation doesn't yet do.
This is a concrete, testable direction for improving retrieval quality:
either reduce hop depth for high-degree nodes, rank/filter retrieved
facts by relevance before passing to the LLM, or add lightweight
community clustering as the original GraphRAG paper does.

**Proposal section this feeds:** Section 2 (directly connects to and critiques the Edge et al. GraphRAG citation), Section 3 (motivates a specific technical improvement: relevance-filtered or clustered subgraph retrieval instead of naive N-hop expansion), Section 4 (concrete evidence of a real limitation found through testing, not assumed)

---
## Finding 5 — Free-tier token limits are a real constraint under heavy testing
**Date:** Aug 13, 2026
**Phase:** Phase 4 (GraphRAG testing)

**What happened:** Hit Groq's free-tier daily token limit (100,000 TPD)
after a day of heavy testing - large documents (Phase 3 graph building
alone made ~91 calls on the 80-page 10-K) plus repeated query testing.
The system failed gracefully with a clear 500 error instead of crashing.

**Why this matters:** A concrete, honest limitation of building on a free
tier - worth noting directly in the proposal rather than glossing over.
Also suggests a concrete engineering improvement: the HOP_DEPTH=2 fix
sends 300+ facts per query, burning tokens fast - reducing hop depth
(the fix I'm testing right now) should also reduce token usage per
query, not just improve answer focus.

**Proposal section this feeds:** Section 4 (honest evidence-based limitations), motivates future work on cost-efficient retrieval (Section 5/Year 2)


----
## Finding 6 — Reducing hop depth doesn't fix hub-node explosion; naive graph traversal needs relevance filtering
**Date:** Aug 13, 2026
**Phase:** Phase 4 (GraphRAG tuning)

**What I did:** Reduced HOP_DEPTH from 2 to 1 and re-ran the identical
query, expecting a smaller, more focused context.

**What happened:** The returned fact count barely shrank, because
"Apple Inc." has hundreds of *direct* (1-hop) connections - it's a hub
even at depth 1. Worse, the two most relevant facts from the depth-2 run
("foreign exchange rate risk affects net sales/gross margins") were lost
entirely at depth 1, since they were reachable only via an intermediate
node. The fix traded away the right answer for a marginal noise reduction.

**Why this matters:** Hop depth is the wrong lever for this problem.
The real fix is relevance filtering - ranking retrieved facts by
similarity to the actual question (e.g., embedding each fact and keeping
only the top N most relevant, similar to a reranking step) rather than
naive graph-distance cutoffs. This is a more precise, testable design
insight than "make the graph smaller."

**Proposal section this feeds:** Section 3 (concrete proposed
improvement: embedding-based relevance filtering of retrieved subgraph
facts before generation - essentially a lightweight reranker), Section 4
(shows iterative, evidence-driven engineering, not just "it worked")

---
## Finding 7 — Relevance filtering fixed retrieval noise, but exposed generation as the real bottleneck
**Date:** Aug 14, 2026
**Phase:** Phase 4 (GraphRAG - relevance filtering fix)

**What I did:** Added embedding-based reranking to GraphRAG - after
graph traversal, rank all retrieved facts by similarity to the question
and keep only the top 25, instead of dumping all 300+ raw facts.

**What happened:** The fix worked precisely as intended - fact count
dropped from 300+ (mostly irrelevant) to 25 (almost entirely on-topic:
risk factors, revenue figures, deferred revenue, R&D expenses). But the
final answer was still "Not found in document," despite the context now
containing exactly the kind of information a human analyst would use to
answer the question.

**Why this matters:** This isolates the failure to the generation step,
not retrieval. I now have direct evidence across three experiments
(Findings 1, 4, 7) that retrieval quality and generation faithfulness
are separable problems - a system can retrieve well and still reason
poorly. This is the central justification for claim-level faithfulness
verification (Phase 5): it's not a nice-to-have add-on, it's addressing
a failure mode that persists even after retrieval is fixed.

**Proposal section this feeds:** Section 1 (sharpens the problem statement precisely), Section 3 (strongest single piece of evidence for why the verifier is the core contribution, not vector RAG or GraphRAG themselves)

---
## Finding 8 — The faithfulness verifier itself has a faithfulness problem (false positive on unsupported causal claim)
**Date:** Aug 14, 2026
**Phase:** Phase 5 (Faithfulness verifier)

**What I did:** Deliberately tested the verifier with an answer containing
an unsupported causal claim ("foreign exchange rate risk affects Apple's
revenue") against evidence that only stated the two facts separately,
with no explicit causal or correlational language connecting them.

**What happened:** The NLI model (cross-encoder/nli-deberta-v3-base)
marked both claims as SUPPORTED with 99%+ confidence - a clear false
positive. It appears to have relied on topical/lexical overlap between
the claim and evidence rather than genuine logical entailment.

**Why this matters:** This is a significant, specific limitation of using
an off-the-shelf, general-purpose NLI model for financial/business
faithfulness checking. General NLI models are trained on everyday
sentence pairs (MNLI/SNLI), not dense financial text with implicit
causal reasoning - they may not distinguish "topically related" from
"logically entailed" in this domain. This directly validates the
proposal's premise that claim-level faithfulness verification for
business documents is an open research problem, not a solved
engineering task - a naive implementation (what I've built) produces
false positives on exactly the kind of unsupported inferential leap that
matters most to catch.

**Proposal section this feeds:** Section 2 (direct evidence supporting
the "has not been systematically evaluated" gap claim), Section 3
(motivates domain-specific NLI fine-tuning or a custom verifier as
future/thesis work, not just applying an off-the-shelf model), Section 5
(Year 2 - concrete plan: fine-tune or evaluate NLI models specifically
on financial claim-evidence pairs, e.g. using FinQA-style data)
--
## Finding 9 — Verifier reliably catches direct contradictions (control test)
**Date:** Aug 14, 2026
**Phase:** Phase 5 (Faithfulness verifier)

**What I did:** Control test - gave the verifier a claim with an
unambiguous, direct numeric contradiction ("$500,000 million" vs. the
evidence's actual "$298,085 million") to check whether it works at all
for the clearest possible case.

**What happened:** Correctly labeled NOT_SUPPORTED / CONTRADICTION with
100% confidence.

**Why this matters:** This confirms the verifier isn't broken outright -
it reliably catches direct, explicit factual contradictions. Combined
with Finding 8, this precisely scopes the actual limitation: the model
handles direct numeric/factual contradiction well, but fails on implied
causal or relational claims where the "contradiction" is really an
unsupported logical leap rather than a conflicting fact. This is a much
sharper, more defensible characterization than "the verifier doesn't
work" - it works for one class of errors and not another, which is
itself a useful, specific research finding.

**Proposal section this feeds:** Section 4 (precise, controlled
before/after evidence - much stronger than an anecdote), Section 3
(scopes exactly what future work needs to improve: causal/inferential
claim verification specifically, not faithfulness checking in general)