"use client";

import React from "react";
import { SentenceDetail } from "@/lib/types";

interface HighlightedEssayProps {
  sentences: SentenceDetail[];
  selectedId: number | null;
  onSelectSentence: (id: number) => void;
}

export const HighlightedEssay: React.FC<HighlightedEssayProps> = ({
  sentences,
  selectedId,
  onSelectSentence,
}) => {
  const getSentenceStyle = (score: number, isSelected: boolean) => {
    let bg = "rgba(16, 185, 129, 0.15)";
    let border = "rgba(16, 185, 129, 0.4)";
    let color = "#6ee7b7";

    if (score >= 0.7) {
      bg = "rgba(239, 68, 68, 0.2)";
      border = "rgba(239, 68, 68, 0.5)";
      color = "#fca5a5";
    } else if (score >= 0.45) {
      bg = "rgba(245, 158, 11, 0.2)";
      border = "rgba(245, 158, 11, 0.5)";
      color = "#fcd34d";
    }

    return {
      backgroundColor: bg,
      borderBottom: `2px solid ${border}`,
      color: "#f1f5f9",
      padding: "0.25rem 0.45rem",
      margin: "0 0.2rem",
      borderRadius: "6px",
      cursor: "pointer",
      outline: isSelected ? `2px solid ${color}` : "none",
      boxShadow: isSelected ? `0 0 12px ${bg}` : "none",
      display: "inline",
      lineHeight: "2.2",
      transition: "all 0.15s ease",
    };
  };

  return (
    <div
      style={{
        background: "var(--bg-card)",
        padding: "1.5rem",
        borderRadius: "var(--radius-lg)",
        border: "1px solid var(--border-color)",
        maxHeight: "420px",
        overflowY: "auto",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
        <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontWeight: 600 }}>
          💡 Click any highlighted sentence to view its specific 9-signal evidence
        </span>
        <div style={{ display: "flex", gap: "0.75rem", fontSize: "0.75rem" }}>
          <span style={{ color: "var(--color-human-text)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-human-border)" }} />
            Human (&lt;45%)
          </span>
          <span style={{ color: "var(--color-mixed-text)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-mixed-border)" }} />
            Mixed (45-70%)
          </span>
          <span style={{ color: "var(--color-ai-text)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-ai-border)" }} />
            AI (&ge;70%)
          </span>
        </div>
      </div>

      <div style={{ fontSize: "0.95rem" }}>
        {sentences.map((s) => (
          <span
            key={s.id}
            style={getSentenceStyle(s.score, selectedId === s.id)}
            onClick={() => onSelectSentence(s.id)}
            title={`Sentence #${s.id + 1} — AI Likelihood: ${Math.round(s.score * 100)}% (Click to inspect)`}
          >
            {s.text}{" "}
          </span>
        ))}
      </div>
    </div>
  );
};
