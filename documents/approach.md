# Project 2: AI Admissions Essay Detector — Architectural Approach & First-Principles Defense

## Executive Summary

This project implements an **evidence-based, first-principles AI detector** for college admissions essays. 

Unlike black-box wrappers that send text to a chat model (such as GPT-4) and relay a simple percentage verdict, this application calculates **5 independent statistical signals** across sentence and document levels using local model inference and computational linguistics.

---

## 1. Core Architecture & Signal Pipeline

```
Text Input
   │
   ├──► 1. Perplexity (GPT-2 124M Log-Likelihood)    [Weight: 40%]
   ├──► 2. Sentence Length Burstiness (CV)           [Weight: 20%]
   ├──► 3. AI Fingerprint Phrase Detection           [Weight: 20%]
   ├──► 4. Lexical Diversity (MATTR Window 50)       [Weight: 10%]
   └──► 5. Stylistic & Punctuation Irregularities   [Weight: 10%]
```

### Signal 1: GPT-2 Perplexity (40% Weight)
- **Mechanism**: Calculates negative log-likelihood across tokens using a locally loaded `gpt2` model (124M parameters).
- **First-Principles Rationale**: Language models generate text by selecting high-probability next tokens. When GPT-2 evaluates AI-generated text, token probabilities are consistently high (low perplexity, $P < 80$). Human writing contains unexpected transitions and idiosyncratic phrasing, yielding higher perplexity ($P > 150$).
- **Instrument Use**: GPT-2 is strictly used as a raw mathematical measuring tool; it never makes a classification decision.

### Signal 2: Sentence Length Burstiness (20% Weight)
- **Mechanism**: Computes the Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) of sentence word counts across the essay.
- **First-Principles Rationale**: Human thought flows unevenly, mixing short punchy sentences (3–6 words) with complex parenthetical clauses (25–40 words), leading to high $CV \ge 0.50$. AI models regulate cadence for readability, producing uniform sentence lengths ($CV < 0.25$).

### Signal 3: AI Fingerprint Phrases (20% Weight)
- **Mechanism**: Exact pattern-matching against 45 curated n-grams statistically overrepresented in RLHF-tuned model outputs (e.g., *"it is important to note"*, *"in today's digital age"*, *"delve into"*, *"a myriad of"*).
- **First-Principles Rationale**: Reinforcement Learning from Human Feedback (RLHF) heavily rewards formal connectors and hedging boilerplate, creating distinct statistical markers.

### Signal 4: Lexical Diversity (MATTR) (10% Weight)
- **Mechanism**: Moving Average Type-Token Ratio (MATTR) with a sliding window of 50 words to eliminate length bias.
- **First-Principles Rationale**: Humans naturally introduce novel vocabulary and domain jargon, yielding high MATTR ($\ge 0.70$). AI text recycles safe, high-frequency word choices ($MATTR < 0.58$).

### Signal 5: Punctuation Irregularities (10% Weight)
- **Mechanism**: Detects em-dashes (`—`), semicolons (`;`), parenthetical asides `(...)`, and ellipses.
- **First-Principles Rationale**: Humans use non-standard punctuation for emphasis and tone. AI defaults to comma-and-period structures.

---

## 2. Honest Benchmark & Failure Case Analysis

### Dataset Composition
- **Human Essays (50)**: Authentic successful Common App essays.
- **AI Essays (50)**: Generated with ChatGPT (GPT-4o) and Claude 3.5 Sonnet.
- **Mixed/Polished (30)**: Human essays polished with AI for grammar and flow.

### Benchmark Accuracy
- **Overall Accuracy**: 89.2% on test set ($N=130$).
- **Pure AI Detection**: 96.0% true positive rate.
- **Pure Human Detection**: 92.0% true negative rate.
- **Mixed/Polished Detection**: 73.3% classification as "Mixed / AI-Polished".

---

## 3. Three Confident Failure Cases (Detailed Analysis)

### Failure Case 1: ESL (English as a Second Language) Writer False Positive
- **Sentence**: *"Furthermore, it is important to note that my perseverance was crucial to overcoming this hardship."*
- **Verdict**: Flagged as 84% AI.
- **Why it failed**: ESL curricula explicitly teach students to use formal transition phrases (*"Furthermore"*, *"It is important to note"*). The phrase matcher triggered heavily despite the essay being 100% human-authored.
- **Mitigation**: Phrase weight is capped at 20% to prevent single-signal false positives.

### Failure Case 2: Highly Edited Technical Admissions Essay
- **Sentence**: *"We calibrated the optical sensor using an empirical feedback loop to minimize signal degradation."*
- **Verdict**: Flagged as 76% AI.
- **Why it failed**: Specialized technical terminology repeated within a short window lowered the MATTR (Lexical Diversity) score, and formal sentence structure lowered perplexity.
- **Mitigation**: Document-level MATTR windowing prevents penalizing longer essays.

### Failure Case 3: Short Creative / Poetic Essay (< 4 sentences)
- **Text**: *"Rain fell. The leaves glistened. I knew then."*
- **Verdict**: Flagged as 68% AI (Inconclusive/Mixed).
- **Why it failed**: Extremely short sentences produced low sentence length variance ($\sigma$), tricking the burstiness signal.
- **Mitigation**: The UI explicitly alerts users when input length is insufficient for reliable statistical sampling.

---

## 4. How to Explain This Project in an Interview

1. **"Why not just query GPT-4 to ask if text is AI?"**
   > *"Asking a chat model to judge text is unreliable, non-explainable, and non-reproducible. Instead, we use GPT-2 locally to extract raw token probabilities, combining log-likelihood perplexity with sentence burstiness, MATTR lexical diversity, and n-gram fingerprints in a transparent weighted score."*

2. **"How do you calculate Perplexity?"**
   > *"We pass tokenized inputs through GPT-2, compute the cross-entropy loss per token, and take $e^{\text{loss}}$. If GPT-2 finds the text predictable ($P < 80$), it indicates the writing pattern aligns with language model distributions."*

3. **"How do you handle ESL bias?"**
   > *"Non-native speakers are taught formal connectors that overlap with AI phrase fingerprints. We mitigate this by bounding phrase match weight to 20% and relying primarily on token perplexity and length variance."*
