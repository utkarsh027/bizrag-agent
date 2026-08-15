"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

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

export default function Home() {
  const [doc, setDoc] = useState<DocStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"query" | "graph-query" | "agent-query" | "compare">("agent-query");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
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

  function FaithfulnessBadge({ f }: { f: Faithfulness }) {
    if (!f) return null;
    const color =
      f.verdict === "HIGH" ? "bg-green-100 text-green-800" :
      f.verdict === "MEDIUM" ? "bg-yellow-100 text-yellow-800" :
      "bg-red-100 text-red-800";
    return (
      <span className={`inline-block px-2 py-1 rounded text-sm font-medium ${color}`}>
        Faithfulness: {f.verdict} ({(f.faithfulness_score * 100).toFixed(0)}%)
      </span>
    );
  }

  function AnswerCard({ r, label }: { r: QueryResult; label: string }) {
    return (
      <div className="border rounded-lg p-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-sm text-gray-500">{label}</span>
          {r.agent_reasoning && (
            <span className="text-xs text-gray-400 italic">{r.agent_reasoning}</span>
          )}
        </div>
        <p className="text-gray-900">{r.answer}</p>
        <FaithfulnessBadge f={r.faithfulness ?? null} />
        <details className="text-sm text-gray-500">
          <summary className="cursor-pointer">
            Sources ({(r.sources || r.graph_facts || []).length})
          </summary>
          <ul className="mt-2 space-y-1 list-disc list-inside">
            {(r.sources || r.graph_facts || []).slice(0, 5).map((s, i) => (
              <li key={i} className="truncate">{s}</li>
            ))}
          </ul>
        </details>
      </div>
    );
  }

  return (
    <main className="max-w-3xl mx-auto p-8 space-y-6">
      <h1 className="text-2xl font-bold">BizRAG-Agent</h1>
      <p className="text-gray-500">Upload a business PDF, ask questions, see faithfulness scores.</p>

      {/* Upload */}
      <div className="border rounded-lg p-4">
        <input type="file" accept=".pdf" onChange={handleUpload} disabled={uploading} />
        {doc && (
          <div className="mt-3 text-sm">
            <p><strong>{doc.filename}</strong> — status: <span className="font-mono">{doc.status}</span></p>
            {doc.status === "ready" && (
              <p className="text-gray-500">
                {doc.chunk_count} chunks · {doc.entity_count} entities extracted
              </p>
            )}
            {doc.status === "failed" && <p className="text-red-600">{doc.error}</p>}
          </div>
        )}
      </div>

      {/* Query */}
      {doc?.status === "ready" && (
        <div className="space-y-3">
          <div className="flex gap-2">
            {(["query", "graph-query", "agent-query", "compare"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1 rounded text-sm border ${
                  mode === m ? "bg-black text-white" : "bg-white text-gray-700"
                }`}
              >
                {m === "query" ? "Vector RAG" : m === "graph-query" ? "GraphRAG" : m === "agent-query" ? "Agent" : "Compare All"}
              </button>
            ))}
          </div>

          <div className="flex gap-2">
            <input
              className="flex-1 border rounded px-3 py-2"
              placeholder="Ask a question about this document..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            />
            <button
              onClick={handleAsk}
              disabled={loading}
              className="bg-black text-white px-4 py-2 rounded disabled:opacity-50"
            >
              {loading ? "Asking..." : "Ask"}
            </button>
          </div>
        </div>
      )}

      {error && <p className="text-red-600">{error}</p>}

      {/* Results */}
      {result && mode !== "compare" && <AnswerCard r={result} label={result.mode} />}

      {result && mode === "compare" && (
        <div className="space-y-4">
          <AnswerCard r={result.vector_rag} label="Vector RAG" />
          <AnswerCard r={result.graph_rag} label="GraphRAG" />
          <AnswerCard r={result.agentic} label={`Agentic (chose: ${result.agentic.agent_strategy})`} />
        </div>
      )}
    </main>
  );
}