"use client";

import React from "react";

interface OverallScoreProps {
  score: number;
  verdict: string;
  totalSentences: number;
  confidenceRange?: { low: number; high: number };
  modelBadge?: string;
}

export const OverallScore: React.FC<OverallScoreProps> = ({
  score,
  verdict,
  totalSentences,
  confidenceRange,
  modelBadge = "9-Signal Platt-Calibrated Ensemble",
}) => {
  const percentage = Math.round(score * 100);

  let badgeClass = "verdict-human";
  let explanation = "Strong presence of natural cognitive burstiness, lexical long-tail richness, and organic syntactic transitions.";
  if (score >= 0.7) {
    badgeClass = "verdict-ai";
    explanation = "High token predictability, uniform syntactic cadence, and strong alignment with RLHF transition patterns.";
  } else if (score >= 0.45) {
    badgeClass = "verdict-mixed";
    explanation = "Hybrid characteristics detected — likely human personal narrative revised or smoothed with AI grammar/flow tools.";
  }

  let barColor = "var(--color-human-border)";
  if (score >= 0.7) barColor = "var(--color-ai-border)";
  else if (score >= 0.45) barColor = "var(--color-mixed-border)";

  const lowPct = confidenceRange ? Math.round(confidenceRange.low * 100) : Math.max(0, percentage - 5);
  const highPct = confidenceRange ? Math.round(confidenceRange.high * 100) : Math.min(100, percentage + 5);

  return (
    <div className="verdict-container">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.4rem" }}>
            <span
              style={{
                fontSize: "0.8rem",
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                fontWeight: 700,
              }}
            >
              Overall Evidentiary Verdict ({totalSentences} sentences)
            </span>
            <span className="model-chip">
              🛡️ {modelBadge}
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "baseline", gap: "1rem", flexWrap: "wrap" }}>
            <span style={{ fontSize: "2.5rem", fontWeight: "800", color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
              {percentage}%
            </span>
            <span style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-secondary)" }}>
              AI Probability
            </span>
            <span className={`verdict-badge ${badgeClass}`}>{verdict}</span>
          </div>
        </div>

        <div className="confidence-pill">
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            95% Calibrated Confidence Range
          </div>
          <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.15rem" }}>
            [{lowPct}% — {highPct}%]
          </div>
        </div>
      </div>

      {/* Progress Bar Meter */}
      <div className="score-meter">
        <div
          className="score-fill"
          style={{
            width: `${percentage}%`,
            backgroundColor: barColor,
          }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.5rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
        <span>0% Organic Human</span>
        <span>45% Mixed / Polished Threshold</span>
        <span>70% Pure AI Threshold</span>
        <span>100% Synthetic AI</span>
      </div>

      <p style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
        {explanation}
      </p>
    </div>
  );
};
