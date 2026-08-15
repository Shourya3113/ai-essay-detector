# Dataset & Evaluation Benchmark

## Sourcing Overview

This benchmark dataset was constructed to evaluate the detector across human-written essays, pure AI-generated essays, and hybrid (human draft + AI polish) essays.

### 1. Human Essays (50 samples)
- **Source**: Publicly available, anonymized successful college admissions essays (e.g., Common App prompts).
- **Characteristics**: Varied sentence lengths, personal anecdotes, non-standard punctuation (em-dashes, parentheticals), domain-specific vocabulary.

### 2. AI-Generated Essays (50 samples)
- **Source**: Generated using ChatGPT (GPT-3.5-Turbo & GPT-4o) and Claude 3.5 Sonnet using standard admissions essay prompts.
- **Characteristics**: High structural uniformity, repetitive transition phrases ("Furthermore", "In conclusion"), low perplexity scores across GPT-2 tokens.

### 3. Mixed / AI-Polished Essays (30 samples)
- **Source**: Human-written essays edited with instructions like "Polish for grammar and flow" using AI tools.
- **Characteristics**: Retains personal narrative details but exhibits low burstiness and standardized sentence structures.

---

## Known Failure Cases & Limitations

1. **ESL (English as a Second Language) Writer Bias**
   - Non-native English writers often rely on formal, structured transition phrases taught in ESL curricula (e.g., "In conclusion", "Furthermore", "It is important to note").
   - This can artificially inflate the **Phrase Match** score. Our model mitigates this by weighing phrases at only 20% and relying primarily on GPT-2 token perplexity.

2. **Overly Formal Academic Essays**
   - Highly polished, formal academic essays without colloquialisms may show low perplexity and low sentence length variance, causing a false positive.

3. **Short Text Inputs (< 3 sentences)**
   - Statistical signals like Burstiness (CV) and Lexical Diversity (MATTR) require sufficient sample size to be meaningful.
