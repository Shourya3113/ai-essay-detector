/**
 * types.ts — TypeScript interface definitions for the 9-Signal AI Essay Detector
 */

export interface PerplexitySignal {
  score: number; // [0, 1] AI likelihood
  raw: number | null;
  cross_entropy?: number | null;
  note: string;
}

export interface BurstinessSignal {
  score: number;
  cv: number | null;
  std_dev: number | null;
  mean_len: number | null;
  note: string;
}

export interface LexicalSignal {
  score: number;
  mattr: number | null;
  hapax_ratio?: number | null;
  unique: number;
  total: number;
  note: string;
}

export interface PhrasesSignal {
  score: number;
  count: number;
  matched: Array<{ phrase: string; reason: string }>;
  note: string;
}

export interface PunctuationSignal {
  score: number;
  count: number;
  markers: string[];
  entropy?: number | null;
  note: string;
}

export interface FunctionWordsSignal {
  score: number;
  rate: number | null;
  entropy: number | null;
  top_words: Array<{ word: string; count: number }>;
  note: string;
}

export interface PosEntropySignal {
  score: number;
  entropy: number | null;
  fluidity: string;
  note: string;
}

export interface ReadabilitySignal {
  score: number;
  fk_grade: number | null;
  reading_ease: number | null;
  variance: number | null;
  note: string;
}

export interface DiscourseMarkersSignal {
  score: number;
  density: number | null;
  markers: Array<{ marker: string; type: string }>;
  note: string;
}

export interface SentenceSignals {
  perplexity: PerplexitySignal;
  phrases: PhrasesSignal;
  punctuation: PunctuationSignal;
  function_words?: FunctionWordsSignal;
  pos_entropy?: PosEntropySignal;
  readability?: ReadabilitySignal;
  discourse_markers?: DiscourseMarkersSignal;
}

export interface SentenceDetail {
  id: number;
  text: string;
  score: number;
  signals: SentenceSignals;
}

export interface DocumentSignalsData {
  burstiness: BurstinessSignal;
  lexical_diversity: LexicalSignal;
  function_words?: FunctionWordsSignal;
  pos_entropy?: PosEntropySignal;
  readability?: ReadabilitySignal;
  discourse_markers?: DiscourseMarkersSignal;
}

export interface AnalysisResult {
  overall_score: number;
  verdict: string;
  confidence_range?: { low: number; high: number };
  model_badge?: string;
  is_fallback?: boolean;
  document_signals: DocumentSignalsData;
  sentences: SentenceDetail[];
}
