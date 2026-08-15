"use client";

import React, { useState } from "react";
import { EssayInput } from "@/components/EssayInput";
import { OverallScore } from "@/components/OverallScore";
import { DocumentSignals } from "@/components/DocumentSignals";
import { HighlightedEssay } from "@/components/HighlightedEssay";
import { SignalBreakdown } from "@/components/SignalBreakdown";
import { AnalysisResult } from "@/lib/types";

export default function Home() {
  const [text, setText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedSentenceId, setSelectedSentenceId] = useState<number | null>(null);
  const [showFallbackBanner, setShowFallbackBanner] = useState<boolean>(true);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setSelectedSentenceId(null);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Analysis failed.");
      }

      const data: AnalysisResult = await res.json();
      setResult(data);
      if (data.sentences && data.sentences.length > 0) {
        setSelectedSentenceId(data.sentences[0].id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not connect to backend server.");
    } finally {
      setLoading(false);
    }
  };

  const selectedSentence =
    result && selectedSentenceId !== null
      ? result.sentences.find((s) => s.id === selectedSentenceId) || null
      : null;

  return (
    <div className="container">
      {/* Header */}
      <header>
        <div className="badge-tag">
          🔬 9-Signal First-Principles Linguistic Engine
        </div>
        <h1>AI Admissions Essay Detector</h1>
        <p>
          Evidence-based statistical and stylometric evaluation measuring token perplexity, sentence burstiness, lexical diversity, and RLHF phrase fingerprints.
        </p>
      </header>

      {/* Fallback Mode Notification Banner */}
      {result?.is_fallback && showFallbackBanner && (
        <div className="fallback-banner">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "1.25rem" }}>⚡</span>
            <div>
              <strong>Running in Lightweight Heuristic Mode</strong>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.15rem" }}>
                Full 9-signal statistical suite active (MATTR, Burstiness, POS Entropy, Readability, N-Grams). To enable live GPT-2 124M neural cross-entropy, launch the Python FastAPI backend on port 8000.
              </div>
            </div>
          </div>
          <button
            type="button"
            className="fallback-dismiss"
            onClick={() => setShowFallbackBanner(false)}
            title="Dismiss notification"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div className="grid-layout">
        {/* Left Column: Essay Input & Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <EssayInput
            text={text}
            setText={setText}
            onAnalyze={handleAnalyze}
            loading={loading}
          />

          {/* Error Toast Notification */}
          {error && (
            <div className="error-toast">
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span>⚠️</span>
                <span>{error}</span>
              </div>
              <button
                type="button"
                className="toast-dismiss"
                onClick={() => setError(null)}
              >
                ✕
              </button>
            </div>
          )}

          {/* Educational Quick Reference Card */}
          <div className="card" style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            <div className="card-title" style={{ fontSize: "1rem" }}>
              <span>Why First-Principles Detection?</span>
            </div>
            <p>
              Unlike commercial LLM wrappers that query GPT-4 to judge student writing, this engine computes <strong>9 reproducible mathematical measurements</strong> across information theory, syntax, and stylometry.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginTop: "0.5rem" }}>
              <div className="feature-pill">1. GPT-2 Perplexity</div>
              <div className="feature-pill">2. Sentence Burstiness (CV)</div>
              <div className="feature-pill">3. MATTR Diversity & Hapax</div>
              <div className="feature-pill">4. AI N-Gram Fingerprints</div>
              <div className="feature-pill">5. Punctuation Uniformity</div>
              <div className="feature-pill">6. Function Word Entropy</div>
              <div className="feature-pill">7. POS Bigram Entropy</div>
              <div className="feature-pill">8. Readability Dynamics</div>
              <div className="feature-pill" style={{ gridColumn: "span 2" }}>
                9. Discourse Marker & Academic Transition Density
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Dynamic Analysis Dashboard */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {result ? (
            <>
              {/* Top Level Calibrated Score */}
              <OverallScore
                score={result.overall_score}
                verdict={result.verdict}
                totalSentences={result.sentences.length}
                confidenceRange={result.confidence_range}
                modelBadge={result.model_badge}
              />

              {/* Document-Level Statistical Profiles */}
              <DocumentSignals signals={result.document_signals} />

              {/* Interactive Sentence Heatmap */}
              <HighlightedEssay
                sentences={result.sentences}
                selectedId={selectedSentenceId}
                onSelectSentence={(id) => setSelectedSentenceId(id)}
              />

              {/* Deep 9-Signal Sentence Diagnostic */}
              <SignalBreakdown sentence={selectedSentence} />
            </>
          ) : (
            <div
              className="card"
              style={{
                justifyContent: "center",
                alignItems: "center",
                minHeight: "420px",
                textAlign: "center",
                color: "var(--text-secondary)",
                padding: "3rem 2rem",
              }}
            >
              <div style={{ fontSize: "3rem", marginBottom: "0.75rem" }}>📊</div>
              <h3 style={{ color: "var(--text-primary)", fontSize: "1.25rem" }}>
                Ready for Evidence Analysis
              </h3>
              <p
                style={{
                  maxWidth: "420px",
                  fontSize: "0.9rem",
                  marginTop: "0.5rem",
                  color: "var(--text-muted)",
                  lineHeight: 1.6,
                }}
              >
                Paste a personal statement or load one of the presets on the left to extract transparent statistical features and inspect sentence-by-sentence evidence.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
