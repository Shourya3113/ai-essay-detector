import { useMemo, useState } from "react";

type FeatureHit = {
  feature: string;
  value: number;
  contribution: number;
  plain: string;
};

type SentenceResult = {
  text: string;
  start: number;
  end: number;
  p_ai: number;
  explanations: FeatureHit[];
};

type DetectResponse = {
  essay_score: { point: number; low: number; high: number; label: string };
  caveat: string;
  sentences: SentenceResult[];
  instruments: Record<string, unknown>;
  model_name: string;
  too_short: boolean;
};

const SAMPLE = `I burned the first tray of baklava on a Thursday that already smelled like bleach. My uncle swore in two languages, then handed me a cracked plastic scraper like that was the whole curriculum. The walnuts had gone bitter. Anyway, I kept the scraper in my backpack for a year. It clicked against my chemistry notes on the bus. People like a version of this story where the kitchen becomes a metaphor for grit. I don't. I was sixteen and slow and I hated being watched. When a college interviewer later asked me about leadership, I talked about the walk-in fridge's bad compressor. She smiled the way adults smile when they think you've said the wrong thing on purpose. I hadn't.`;

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function flagColor(p: number): string {
  const t = Math.min(1, Math.max(0, (p - 0.35) / 0.5));
  const a = 0.08 + t * 0.45;
  return `rgba(180, 83, 9, ${a})`;
}

export default function App() {
  const [text, setText] = useState(SAMPLE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [active, setActive] = useState<number | null>(null);

  const trained = result?.instruments?.trained_classifier === true;
  const gpt2 = result?.instruments?.gpt2 === true;

  const selected = useMemo(() => {
    if (result == null || active == null) return null;
    return result.sentences[active] ?? null;
  }, [result, active]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setActive(null);
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || "Request failed");
      }
      setResult(body as DetectResponse);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-6 py-8">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-stone-500">
              Admissions tool · not an honor-code verdict
            </p>
            <h1 className="font-serif text-4xl tracking-tight">EssayTrace</h1>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-stone-600">
              Highlights <em>where</em> an essay looks unusually smooth, uniform,
              or formulaic — from measured features, not from asking a chatbot
              “is this AI?”
            </p>
          </div>
          <p className="max-w-xs text-right text-xs leading-relaxed text-stone-500">
            DistilGPT-2 supplies token surprisal locally. No instruction-tuned
            model is allowed to render the score.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-8 px-6 py-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section>
          <form onSubmit={onSubmit} className="space-y-3">
            <label className="text-xs uppercase tracking-widest text-stone-500">
              Paste an essay
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={12}
              className="w-full resize-y rounded-sm border border-rule bg-white/70 p-4 font-serif text-[17px] leading-7 outline-none ring-flag/30 focus:ring-2"
            />
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={loading}
                className="rounded-sm bg-ink px-4 py-2 text-sm text-paper disabled:opacity-50"
              >
                {loading ? "Measuring…" : "Measure sentences"}
              </button>
              <button
                type="button"
                className="text-sm text-stone-500 underline-offset-2 hover:underline"
                onClick={() => {
                  setText(SAMPLE);
                  setResult(null);
                }}
              >
                Reset sample
              </button>
            </div>
          </form>

          {error && (
            <p className="mt-4 border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
              {error}
            </p>
          )}

          {result && (
            <div className="mt-8">
              <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
                <h2 className="font-serif text-2xl">Marked-up essay</h2>
                <p className="text-xs text-stone-500">
                  Color intensity follows sentence-level P(AI). Click a sentence.
                </p>
              </div>
              <article className="rounded-sm border border-rule bg-white/80 p-6 font-serif text-[18px] leading-8">
                {result.sentences.map((s, i) => (
                  <span
                    key={`${s.start}-${i}`}
                    onClick={() => setActive(i)}
                    onMouseEnter={() => setActive(i)}
                    style={{ backgroundColor: flagColor(s.p_ai) }}
                    className={`cursor-pointer rounded-sm px-[1px] transition ${
                      active === i ? "ring-2 ring-flag" : ""
                    }`}
                    title={pct(s.p_ai)}
                  >
                    {s.text}{" "}
                  </span>
                ))}
              </article>
            </div>
          )}
        </section>

        <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          <div className="border border-rule bg-white/60 p-4">
            <h2 className="text-xs uppercase tracking-widest text-stone-500">
              Essay-level signal
            </h2>
            {result ? (
              <>
                <p className="mt-3 font-serif text-3xl tabular-nums">
                  {pct(result.essay_score.low)}–{pct(result.essay_score.high)}
                </p>
                <p className="mt-1 text-sm text-stone-600">
                  Calibrated range · point {pct(result.essay_score.point)}
                </p>
                <p className="mt-2 text-sm font-medium">{result.essay_score.label}</p>
                <p className="mt-3 border-t border-rule pt-3 text-xs leading-relaxed text-stone-600">
                  {result.caveat}
                </p>
                <ul className="mt-3 space-y-1 text-[11px] text-stone-500">
                  <li>Classifier: {result.model_name}</li>
                  <li>Trained artifact: {trained ? "yes" : "heuristic fallback"}</li>
                  <li>DistilGPT-2 NLL: {gpt2 ? "on" : "char-trigram fallback"}</li>
                </ul>
              </>
            ) : (
              <p className="mt-3 text-sm text-stone-500">
                Run a measurement to see a range, never a lone percentage.
              </p>
            )}
          </div>

          <div className="border border-rule bg-white/60 p-4 min-h-[14rem]">
            <h2 className="text-xs uppercase tracking-widest text-stone-500">
              Why this sentence
            </h2>
            {selected ? (
              <>
                <p className="mt-3 font-serif text-sm leading-6">“{selected.text}”</p>
                <p className="mt-2 text-xs tabular-nums text-stone-500">
                  Sentence signal {pct(selected.p_ai)}
                </p>
                <ol className="mt-3 space-y-3">
                  {selected.explanations.map((h) => (
                    <li key={h.feature} className="text-sm leading-6">
                      {h.plain}
                    </li>
                  ))}
                </ol>
              </>
            ) : (
              <p className="mt-3 text-sm text-stone-500">
                Hover or click a highlighted sentence for the top features, in
                plain language.
              </p>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
