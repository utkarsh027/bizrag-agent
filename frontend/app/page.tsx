"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DocStatus = {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count?: number;
  entity_count?: number;
  error?: string;
};

type Claim = { claim: string; verdict: string; confidence: number };
type Faithfulness = { faithfulness_score: number; verdict: string; claims: Claim[] } | null;

type QueryResult = {
  answer: string;
  sources?: string[];
  graph_facts?: string[];
  mode: string;
  agent_strategy?: string;
  agent_reasoning?: string;
  faithfulness?: Faithfulness;
};

const STAGES = ["extracting_text", "chunking", "embedding", "building_graph", "ready"];
const STAGE_LABELS: Record<string, string> = {
  extracting_text: "Extract",
  chunking: "Chunk",
  embedding: "Embed",
  building_graph: "Graph",
  ready: "Ready",
};

const MODES = [
  { key: "query", label: "VEC", full: "Vector RAG" },
  { key: "graph-query", label: "GRPH", full: "GraphRAG" },
  { key: "agent-query", label: "AGT", full: "Agent" },
  { key: "compare", label: "CMP", full: "Compare All" },
] as const;

function PipelineStepper({ status }: { status: string }) {
  if (status === "failed") {
    return <p className="font-mono text-xs text-[var(--rose)] mt-3">✕ FAILED — see error above</p>;
  }
  const currentIdx = STAGES.indexOf(status);
  return (
    <div className="mt-3">
      <div className="flex items-center gap-1">
        {STAGES.map((s, i) => {
          const done = currentIdx > i || status === "ready";
          const active = currentIdx === i && status !== "ready";
          return (
            <div
              key={s}
              className={`h-1.5 flex-1 rounded-full transition-colors duration-500 ${
                done ? "bg-[var(--teal)]" : active ? "bg-[var(--signal)] pulse-dot" : "bg-[var(--shell-line)]"
              }`}
            />
          );
        })}
      </div>
      <div className="flex justify-between mt-1.5 font-mono text-[10px] uppercase tracking-wider text-[var(--muted)]">
        {STAGES.map((s) => (
          <span key={s}>{STAGE_LABELS[s]}</span>
        ))}
      </div>
    </div>
  );
}

function LedgerMeter({ f }: { f: Faithfulness }) {
  if (!f) return null;
  const pct = Math.round(f.faithfulness_score * 100);
  const color = f.verdict === "HIGH" ? "var(--teal)" : f.verdict === "MEDIUM" ? "var(--amber)" : "var(--rose)";
  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--paper-muted)]">Faithfulness</span>
        <span className="font-mono text-[10px] uppercase tracking-wider font-semibold" style={{ color }}>
          {f.verdict} · {pct}%
        </span>
      </div>
      <div className="relative h-2 rounded-full overflow-hidden" style={{ background: "var(--paper-line)" }}>
        <div className="meter-fill h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
        {[25, 50, 75].map((t) => (
          <div key={t} className="absolute top-0 h-full w-px" style={{ left: `${t}%`, background: "var(--paper)" }} />
        ))}
      </div>
    </div>
  );
}

function AnswerCard({ r, label }: { r: QueryResult; label: string }) {
  return (
    <div
      className="animate-fade-up rounded-xl p-5 shadow-sm"
      style={{ background: "var(--paper)", color: "var(--paper-text)", border: "1px solid var(--paper-line)" }}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-wider font-semibold" style={{ color: "var(--paper-muted)" }}>
          {label}
        </span>
      </div>
      {r.agent_reasoning && (
        <p className="text-xs italic mt-2 mb-1" style={{ color: "var(--paper-muted)" }}>{r.agent_reasoning}</p>
      )}
      <p className="leading-relaxed mt-2">{r.answer || "No answer returned."}</p>
      <LedgerMeter f={r.faithfulness ?? null} />
      <details className="mt-4 text-sm group">
        <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider font-semibold" style={{ color: "var(--paper-muted)" }}>
          Sources ({(r.sources || r.graph_facts || []).length})
        </summary>
        <ul className="mt-2 space-y-1.5">
          {(r.sources || r.graph_facts || []).slice(0, 6).map((s, i) => (
            <li
              key={i}
              className="font-mono text-xs truncate border-l-2 pl-2"
              style={{ borderColor: "var(--paper-line)", color: "var(--paper-muted)" }}
            >
              {s}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function GraphViewer({ docId }: { docId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let graphInstance: any;

    (async () => {
      const ForceGraph = (await import("force-graph")).default;
      const res = await fetch(`${API_BASE}/graph/${docId}`);
      if (!res.ok) return;
      const data = await res.json();

      graphInstance = new ForceGraph(containerRef.current!)
        .graphData(data)
        .backgroundColor("transparent")
        .nodeLabel("id")
        .nodeColor(() => "#4c8dff")
        .nodeRelSize(3)
        .linkColor(() => "rgba(136, 146, 164, 0.35)")
        .linkLabel("relation")
        .linkDirectionalArrowLength(3)
        .linkDirectionalArrowColor(() => "#4c8dff")
        .width(containerRef.current!.clientWidth)
        .height(380);
    })();

    return () => { graphInstance?._destructor?.(); };
  }, [docId]);

  return <div ref={containerRef} className="rounded-lg overflow-hidden" style={{ background: "var(--shell-surface-2)" }} />;
}

export default function Home() {
  const [doc, setDoc] = useState<DocStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"query" | "graph-query" | "agent-query" | "compare">("agent-query");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [showGraph, setShowGraph] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError("");
    setDoc(null);
    setResult(null);
    setShowGraph(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
      const data = await res.json();

      pollRef.current = setInterval(async () => {
        const statusRes = await fetch(`${API_BASE}/documents/${data.doc_id}`);
        if (!statusRes.ok) return;
        const statusData: DocStatus = await statusRes.json();
        setDoc(statusData);
        if (statusData.status === "ready" || statusData.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setUploading(false);
        }
      }, 2000);
    } catch (err: any) {
      setError(err.message);
      setUploading(false);
    }
  }

  async function handleAsk() {
    if (!doc || !query.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, doc_id: doc.doc_id }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Query failed");
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const ready = doc?.status === "ready";

  return (
    <main className="min-h-screen">
      {/* Top bar */}
      <div className="border-b" style={{ borderColor: "var(--shell-line)" }}>
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-display font-semibold text-lg tracking-tight">BizRAG</span>
            <span className="font-mono text-xs text-[var(--muted)]">// AGENT</span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-[var(--muted)]">
            <span className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-[var(--teal)]" : "bg-[var(--shell-line)]"}`} />
            {ready ? "Document Ready" : "Awaiting Document"}
          </div>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-10 space-y-6">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight">Faithful retrieval for business documents</h1>
          <p className="text-[var(--muted)] mt-1 text-sm">
            Upload a filing or report. Ask a question. Every answer ships with a measured confidence score, not just a guess.
          </p>
        </div>

        {/* Upload */}
        <div
          className="rounded-xl border-2 border-dashed p-6 transition-colors"
          style={{ borderColor: "var(--shell-line)", background: "var(--shell-surface)" }}
        >
          <label className="flex items-center gap-3 cursor-pointer">
            <span
              className="font-mono text-xs uppercase tracking-wider px-3 py-1.5 rounded-md"
              style={{ background: "var(--shell-surface-2)", color: "var(--ink-text)" }}
            >
              Choose PDF
            </span>
            <span className="text-sm text-[var(--muted)]">
              {doc ? doc.filename : "No document loaded — upload one to begin"}
            </span>
            <input type="file" accept=".pdf" onChange={handleUpload} disabled={uploading} className="hidden" />
          </label>

          {doc && (
            <div className="mt-4">
              <div className="flex items-center justify-between font-mono text-xs">
                <span className="font-semibold">{doc.filename}</span>
                <span className="text-[var(--muted)] uppercase">{doc.status}</span>
              </div>
              <PipelineStepper status={doc.status} />
              {doc.status === "ready" && (
                <p className="text-xs text-[var(--muted)] mt-2 font-mono">
                  {doc.chunk_count} chunks · {doc.entity_count} entities extracted
                </p>
              )}
              {doc.status === "failed" && (
                <p className="text-xs text-[var(--rose)] mt-2">{doc.error}</p>
              )}
            </div>
          )}
        </div>

        {/* Mode + query */}
        {ready && (
          <div className="space-y-3">
            <div className="flex gap-2">
              {MODES.map((m) => (
                <button
                  key={m.key}
                  onClick={() => { setMode(m.key); setResult(null); setError(""); }}
                  title={m.full}
                  className="focus-ring font-mono text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border transition-colors"
                  style={
                    mode === m.key
                      ? { background: "var(--signal)", borderColor: "var(--signal)", color: "#fff" }
                      : { background: "var(--shell-surface)", borderColor: "var(--shell-line)", color: "var(--muted)" }
                  }
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="flex gap-2">
              <input
                className="focus-ring flex-1 rounded-md border px-3 py-2.5 text-sm"
                style={{ background: "var(--shell-surface)", borderColor: "var(--shell-line)", color: "var(--ink-text)" }}
                placeholder="Ask a question about this document…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              />
              <button
                onClick={handleAsk}
                disabled={loading}
                className="focus-ring rounded-md px-5 text-sm font-medium transition-opacity disabled:opacity-50 flex items-center gap-2"
                style={{ background: "var(--signal)", color: "#fff" }}
              >
                {loading && <span className="spin-slow h-3 w-3 border-2 border-white/40 border-t-white rounded-full" />}
                {loading ? "Asking" : "Ask"}
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-md p-3 text-sm font-mono" style={{ background: "rgba(242,100,92,0.1)", color: "var(--rose)" }}>
            {error}
          </div>
        )}

        {/* Results */}
        {result && mode !== "compare" && <AnswerCard r={result} label={result.mode} />}

        {result && mode === "compare" && (
          <div className="space-y-4">
            <AnswerCard r={result.vector_rag} label="Vector RAG" />
            <AnswerCard r={result.graph_rag} label="GraphRAG" />
            <AnswerCard r={result.agentic} label={`Agentic → ${result.agentic.agent_strategy}`} />
          </div>
        )}

        {/* Graph */}
        {ready && (
          <div className="rounded-xl border p-4" style={{ borderColor: "var(--shell-line)", background: "var(--shell-surface)" }}>
            <button
              onClick={() => setShowGraph(!showGraph)}
              className="focus-ring font-mono text-xs uppercase tracking-wider text-[var(--muted)] flex items-center gap-2"
            >
              <span className={`transition-transform ${showGraph ? "rotate-90" : ""}`}>▸</span>
              Knowledge Graph
            </button>
            {showGraph && (
              <div className="mt-3 animate-fade-up">
                <GraphViewer docId={doc!.doc_id} />
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}