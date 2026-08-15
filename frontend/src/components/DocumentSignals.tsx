"use client";

import React from "react";
import { DocumentSignalsData } from "@/lib/types";

interface DocumentSignalsProps {
  signals: DocumentSignalsData;
}

export const DocumentSignals: React.FC<DocumentSignalsProps> = ({ signals }) => {
  const {
    burstiness,
    lexical_diversity,
    function_words,
    pos_entropy,
    readability,
    discourse_markers,
  } = signals;

  const renderBadge = (score: number) => {
    if (score >= 0.65) return <span className="signal-badge-ai">AI-leaning</span>;
    if (score <= 0.35) return <span className="signal-badge-human">Human-leaning</span>;
    return <span className="signal-badge-mixed">Balanced</span>;
  };

  const renderProgressBar = (score: number) => {
    const pct = Math.round(score * 100);
    let color = "var(--color-human-border)";
    if (score >= 0.65) color = "var(--color-ai-border)";
    else if (score >= 0.35) color = "var(--color-mixed-border)";

    return (
      <div className="signal-bar-track">
        <div
          className="signal-bar-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)" }}>
          Document-Level Statistical Profiles
        </h3>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          Global Macro-Linguistic Invariants
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
        {/* Burstiness Metric Card */}
        <div className="doc-card">
          <div className="doc-card-header">
            <div className="doc-card-title">
              Sentence Burstiness (CV)
              <span className="tooltip-icon" title="Coefficient of Variation (CV = σ / μ). Human thought mixes punchy short sentences with sprawling multi-clause sentences. AI regularizes cadence.">
                ⓘ
              </span>
            </div>
            {renderBadge(burstiness.score)}
          </div>
          {renderProgressBar(burstiness.score)}
          <div className="doc-card-metric">
            CV: <strong>{burstiness.cv !== null ? burstiness.cv : "N/A"}</strong> | Mean:{" "}
            <strong>{burstiness.mean_len ?? 0}w</strong> | σ:{" "}
            <strong>{burstiness.std_dev ?? 0}</strong>
          </div>
          <p className="doc-card-note">{burstiness.note}</p>
        </div>

        {/* Lexical Diversity Metric Card */}
        <div className="doc-card">
          <div className="doc-card-header">
            <div className="doc-card-title">
              Lexical Diversity (MATTR)
              <span className="tooltip-icon" title="Moving Average Type-Token Ratio across 50-word sliding windows + Hapax Legomena ratio. Eliminates essay length bias.">
                ⓘ
              </span>
            </div>
            {renderBadge(lexical_diversity.score)}
          </div>
          {renderProgressBar(lexical_diversity.score)}
          <div className="doc-card-metric">
            MATTR: <strong>{lexical_diversity.mattr !== null ? lexical_diversity.mattr : "N/A"}</strong> |{" "}
            {lexical_diversity.unique} unique / {lexical_diversity.total} words
          </div>
          <p className="doc-card-note">{lexical_diversity.note}</p>
        </div>

        {/* POS Transition Entropy Card */}
        {pos_entropy && (
          <div className="doc-card">
            <div className="doc-card-header">
              <div className="doc-card-title">
                POS Transition Entropy
                <span className="tooltip-icon" title="Entropy of Part-of-Speech bigram Markov transitions. Measures grammatical structural dynamism vs formulaic SVO templates.">
                  ⓘ
                </span>
              </div>
              {renderBadge(pos_entropy.score)}
            </div>
            {renderProgressBar(pos_entropy.score)}
            <div className="doc-card-metric">
              Entropy H: <strong>{pos_entropy.entropy ?? "N/A"}</strong> | Fluidity:{" "}
              <strong>{pos_entropy.fluidity}</strong>
            </div>
            <p className="doc-card-note">{pos_entropy.note}</p>
          </div>
        )}

        {/* Readability & Cross-Sentence Variance Card */}
        {readability && (
          <div className="doc-card">
            <div className="doc-card-header">
              <div className="doc-card-title">
                Readability Dynamics (FKGL)
                <span className="tooltip-icon" title="Flesch-Kincaid Grade Level & cross-sentence variance (σ²). AI maintains flat difficulty; human prose fluctuates between simple reflection and dense analysis.">
                  ⓘ
                </span>
              </div>
              {renderBadge(readability.score)}
            </div>
            {renderProgressBar(readability.score)}
            <div className="doc-card-metric">
              Grade: <strong>{readability.fk_grade ?? "N/A"}</strong> | Reading Ease:{" "}
              <strong>{readability.reading_ease ?? "N/A"}</strong> | σ²:{" "}
              <strong>{readability.variance ?? "N/A"}</strong>
            </div>
            <p className="doc-card-note">{readability.note}</p>
          </div>
        )}

        {/* Function Words Stylometry Card */}
        {function_words && (
          <div className="doc-card">
            <div className="doc-card-header">
              <div className="doc-card-title">
                Function Word Stylometry
                <span className="tooltip-icon" title="Subconscious closed-class word frequencies (Mosteller & Wallace). AI generates unnaturally flat, predictable function-word rates.">
                  ⓘ
                </span>
              </div>
              {renderBadge(function_words.score)}
            </div>
            {renderProgressBar(function_words.score)}
            <div className="doc-card-metric">
              Rate: <strong>{function_words.rate !== null ? `${(function_words.rate * 100).toFixed(1)}%` : "N/A"}</strong> | H:{" "}
              <strong>{function_words.entropy ?? "N/A"}</strong>
            </div>
            <p className="doc-card-note">{function_words.note}</p>
          </div>
        )}

        {/* Discourse Transitions Card */}
        {discourse_markers && (
          <div className="doc-card">
            <div className="doc-card-header">
              <div className="doc-card-title">
                Discourse Transition Density
                <span className="tooltip-icon" title="Frequency of explicit academic connectors (furthermore, in conclusion). AI over-saturates transitions to force coherence.">
                  ⓘ
                </span>
              </div>
              {renderBadge(discourse_markers.score)}
            </div>
            {renderProgressBar(discourse_markers.score)}
            <div className="doc-card-metric">
              Density: <strong>{discourse_markers.density ?? 0} markers/sent</strong> | Total:{" "}
              <strong>{discourse_markers.markers.length}</strong>
            </div>
            <p className="doc-card-note">{discourse_markers.note}</p>
          </div>
        )}
      </div>
    </div>
  );
};
