"use client";

import React from "react";

interface EssayInputProps {
  text: string;
  setText: (text: string) => void;
  onAnalyze: () => void;
  loading: boolean;
}

export const SAMPLE_ESSAYS = {
  human: `When I was twelve, my grandfather taught me how to restore vintage bicycles in his dusty garage. He never used instruction manuals or digital tools; instead, he relied on intuition, grease-stained hands, and decades of trial and error. That small, dimly lit workshop became my first laboratory. I didn’t just learn how to realign gear shifters or patch rubber inner tubes—I learned how to diagnose complex physical systems by listening to their quietest friction points. That early fascination with mechanical problem-solving eventually evolved into my passion for computer science and hardware engineering.`,

  ai: `In conclusion, pursuing higher education at this esteemed university represents a crucial milestone in my academic and professional journey. Furthermore, it is important to note that the multifaceted curriculum aligns perfectly with my long-term career aspirations. In today's rapidly evolving digital age, navigating the complexities of modern software engineering requires a deep dedication to continuous learning and innovation. Ultimately, my diverse background and unwavering enthusiasm make me a strong candidate who will make a meaningful contribution to the campus community.`,

  mixed: `Throughout my high school years, I spent endless hours organizing community food drives. It is important to note that coordinating volunteers across three districts presented a variety of challenges. However, by leveraging modern digital spreadsheets and communication tools, we managed to streamline our distribution process. Furthermore, this experience taught me how essential leadership and adaptability are when managing complex logistical operations under pressure.`,

  esl: `When I moved from Seoul to Chicago at age fifteen, the language barrier was my greatest challenge. In the beginning, I hesitated to speak in class because I feared making grammatical mistakes. Furthermore, it is important to note that participating in the school robotics team gave me a universal language of code and mathematics. In conclusion, through hard work and collaboration with my teammates, I learned that communication is built on shared goals rather than perfect fluency.`,

  adversarial: `Running at 5am in the freezing rain is honestly the worst thing ever, ngl. My legs felt like lead weights. But I kept pushing—step by step, breath by breath—cuz giving up just wasn't an option. It was a tough lesson in discipline, but totaly worth it.`,
};

export const EssayInput: React.FC<EssayInputProps> = ({
  text,
  setText,
  onAnalyze,
  loading,
}) => {
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <div className="card">
      <div className="card-title">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span>Input Essay</span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "normal" }}>
            (College Admissions / Personal Statement)
          </span>
        </div>
        <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: "normal" }}>
          {wordCount} words ({text.length} chars)
        </span>
      </div>

      <textarea
        placeholder="Paste a college admissions essay or personal statement here (minimum 10 characters)..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={loading}
      />

      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
          <div className="sample-buttons">
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", alignSelf: "center", fontWeight: 600 }}>
              Samples:
            </span>
            <button
              type="button"
              className="btn-sample"
              onClick={() => setText(SAMPLE_ESSAYS.human)}
              disabled={loading}
            >
              🌿 Human Narrative
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => setText(SAMPLE_ESSAYS.ai)}
              disabled={loading}
            >
              🤖 Pure AI Cliché
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => setText(SAMPLE_ESSAYS.mixed)}
              disabled={loading}
            >
              🔀 AI-Polished Hybrid
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => setText(SAMPLE_ESSAYS.esl)}
              disabled={loading}
            >
              🌏 ESL Student
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => setText(SAMPLE_ESSAYS.adversarial)}
              disabled={loading}
            >
              ⚡ Adversarial AI
            </button>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "0.5rem", borderTop: "1px solid var(--border-color)" }}>
          <button
            type="button"
            className="btn-clear"
            onClick={() => setText("")}
            disabled={loading || !text}
          >
            Clear Text
          </button>

          <button
            type="button"
            className="btn-primary"
            onClick={onAnalyze}
            disabled={loading || text.trim().length < 10}
          >
            {loading ? (
              <>
                <span className="spinner" />
                <span>Extracting 9 Signals...</span>
              </>
            ) : (
              <>
                <span>Run 9-Signal Analysis</span>
                <span>→</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
