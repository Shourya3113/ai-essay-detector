"use client";

import React from "react";
import { SentenceDetail } from "@/lib/types";

interface SignalBreakdownProps {
  sentence: SentenceDetail | null;
}

export const SignalBreakdown: React.FC<SignalBreakdownProps> = ({ sentence }) => {
  if (!sentence) {
    return (
      <div
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-lg)",
          padding: "2rem",
          textAlign: "center",
          color: "var(--text-muted)",
        }}
      >
        <div style={{ fontSize: "1.8rem", marginBottom: "0.5rem" }}>👆</div>
        <p style={{ fontWeight: 500, color: "var(--text-secondary)" }}>
          Select any highlighted sentence in the essay above to inspect its 9-signal evidence breakdown.
        </p>
      </div>
    );
  }

  const {
    perplexity,
    phrases,
    punctuation,
    function_words,
    pos_entropy,
    readability,
    discourse_markers,
  } = sentence.signals;

  const renderBadge = (score: number) => {
    if (score >= 0.65) {
      return <span className="signal-badge-ai">AI-leaning</span>;
    }
    if (score <= 0.35) {
      return <span className="signal-badge-human">Human-leaning</span>;
    }
    return <span className="signal-badge-mixed">Neutral / Mixed</span>;
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
    <div
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-color)",
        borderRadius: "var(--radius-lg)",
        padding: "1.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "1.25rem",
      }}
    >
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span
            style={{
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              fontWeight: 700,
              letterSpacing: "0.05em",
            }}
          >
            Sentence #{sentence.id + 1} Deep Diagnostic
          </span>
          <span
            style={{
              fontSize: "0.9rem",
              fontWeight: 700,
              color:
                sentence.score >= 0.7
                  ? "var(--color-ai-border)"
                  : sentence.score >= 0.45
                  ? "var(--color-mixed-border)"
                  : "var(--color-human-border)",
            }}
          >
            {Math.round(sentence.score * 100)}% AI Likelihood
          </span>
        </div>
        <blockquote
          style={{
            fontSize: "0.95rem",
            color: "var(--text-primary)",
            borderLeft: "3px solid var(--accent-blue)",
            paddingLeft: "0.85rem",
            margin: "0.75rem 0 0.5rem 0",
            fontStyle: "italic",
            lineHeight: 1.6,
          }}
        >
          "{sentence.text}"
        </blockquote>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "-0.5rem" }}>
        <h4 style={{ fontSize: "0.95rem", color: "var(--text-primary)", fontWeight: 600 }}>
          9-Signal Multi-Dimensional Evaluation
        </h4>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          (Hover over ⓘ for feature definitions)
        </span>
      </div>

      {/* Grid of 9 Signal Cards */}
      <div className="signal-grid">
        {/* Signal 1: GPT-2 Perplexity */}
        <div className="signal-card">
          <div className="signal-card-header">
            <div className="signal-card-title">
              1. Token Perplexity (PPL)
              <span className="tooltip-icon" title="Measures raw negative log-likelihood of token transitions using GPT-2. AI text follows predictable low-perplexity valleys.">
                ⓘ
              </span>
            </div>
            {renderBadge(perplexity.score)}
          </div>
          {renderProgressBar(perplexity.score)}
          <div className="signal-card-metric">
            Raw PPL: <strong>{perplexity.raw !== null ? perplexity.raw : "N/A"}</strong>
            {perplexity.cross_entropy && ` | H: ${perplexity.cross_entropy} bits`}
          </div>
          <div className="signal-card-note">{perplexity.note}</div>
        </div>

        {/* Signal 2: AI Fingerprint Phrases */}
        <div className="signal-card">
          <div className="signal-card-header">
            <div className="signal-card-title">
              2. AI Phrase Patterns
              <span className="tooltip-icon" title="Matches 65+ overrepresented RLHF n-gram clichés (e.g. 'it is important to note', 'delve into', 'multifaceted').">
                ⓘ
              </span>
            </div>
            {renderBadge(phrases.score)}
          </div>
          {renderProgressBar(phrases.score)}
          <div className="signal-card-metric">
            Matches: <strong>{phrases.count}</strong>
          </div>
          {phrases.matched.length > 0 ? (
            <ul className="signal-card-list">
              {phrases.matched.map((m, i) => (
                <li key={i}>
                  <em>"{m.phrase}"</em> — <span style={{ color: "var(--text-muted)" }}>{m.reason}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="signal-card-note">No formulaic AI clichés detected.</div>
          )}
        </div>

        {/* Signal 3: Punctuation & Expressive Rhythm */}
        <div className="signal-card">
          <div className="signal-card-header">
            <div className="signal-card-title">
              3. Punctuation Markers
              <span className="tooltip-icon" title="Detects expressive human punctuation (em-dashes —, semicolons ;, parentheticals, ellipses) vs rigid AI periods/commas.">
                ⓘ
              </span>
            </div>
            {renderBadge(punctuation.score)}
          </div>
          {renderProgressBar(punctuation.score)}
          <div className="signal-card-metric">
            Markers: <strong>{punctuation.count}</strong>
          </div>
          {punctuation.markers.length > 0 ? (
            <div className="signal-card-markers">
              {punctuation.markers.map((m, i) => (
                <span key={i} className="marker-tag">{m}</span>
              ))}
            </div>
          ) : (
            <div className="signal-card-note">Standard formal punctuation structure.</div>
          )}
        </div>

        {/* Signal 4: Function Word Distribution */}
        {function_words && (
          <div className="signal-card">
            <div className="signal-card-header">
              <div className="signal-card-title">
                4. Function Word Entropy
                <span className="tooltip-icon" title="Mosteller-Wallace stylometry measuring closed-class word frequencies (the, in, of, that). AI exhibits unnaturally flat distributions.">
                  ⓘ
                </span>
              </div>
              {renderBadge(function_words.score)}
            </div>
            {renderProgressBar(function_words.score)}
            <div className="signal-card-metric">
              Rate: <strong>{function_words.rate !== null ? `${(function_words.rate * 100).toFixed(0)}%` : "N/A"}</strong>
              {function_words.entropy && ` | H: ${function_words.entropy}`}
            </div>
            <div className="signal-card-note">{function_words.note}</div>
          </div>
        )}

        {/* Signal 5: POS Bigram Transition Entropy */}
        {pos_entropy && (
          <div className="signal-card">
            <div className="signal-card-header">
              <div className="signal-card-title">
                5. POS Transition Entropy
                <span className="tooltip-icon" title="Measures grammatical sequence dynamism. Human writing uses rich clausal transitions; AI relies on standard SVO templates.">
                  ⓘ
                </span>
              </div>
              {renderBadge(pos_entropy.score)}
            </div>
            {renderProgressBar(pos_entropy.score)}
            <div className="signal-card-metric">
              Entropy: <strong>{pos_entropy.entropy ?? "N/A"}</strong> | Fluidity: <strong>{pos_entropy.fluidity}</strong>
            </div>
            <div className="signal-card-note">{pos_entropy.note}</div>
          </div>
        )}

        {/* Signal 6: Readability & Grade Level */}
        {readability && (
          <div className="signal-card">
            <div className="signal-card-header">
              <div className="signal-card-title">
                6. Readability Index (FKGL)
                <span className="tooltip-icon" title="Calculates Flesch-Kincaid Grade Level and Reading Ease. AI maintains homogeneous grade levels across all sentences.">
                  ⓘ
                </span>
              </div>
              {renderBadge(readability.score)}
            </div>
            {renderProgressBar(readability.score)}
            <div className="signal-card-metric">
              Grade: <strong>{readability.fk_grade ?? "N/A"}</strong> | Ease: <strong>{readability.reading_ease ?? "N/A"}</strong>
            </div>
            <div className="signal-card-note">{readability.note}</div>
          </div>
        )}

        {/* Signal 7: Discourse Markers & Transitions */}
        {discourse_markers && (
          <div className="signal-card">
            <div className="signal-card-header">
              <div className="signal-card-title">
                7. Discourse Transitions
                <span className="tooltip-icon" title="Evaluates density of explicit transitional connectors (furthermore, moreover, consequently). AI over-indexes to force coherence.">
                  ⓘ
                </span>
              </div>
              {renderBadge(discourse_markers.score)}
            </div>
            {renderProgressBar(discourse_markers.score)}
            <div className="signal-card-metric">
              Found: <strong>{discourse_markers.markers.length} marker(s)</strong>
            </div>
            <div className="signal-card-note">{discourse_markers.note}</div>
          </div>
        )}
      </div>
    </div>
  );
};
