# EssayTrace

Sentence-level signals for **likely AI-written or AI-polished** passages in college admissions essays — with a range, not a fake-precise percentage, and with a plain-English *why* for every flagged sentence.

This is not a chatbot wrapper. The live app **never** sends essay text to GPT-4, Claude, Grok, or any instruction-tuned model to ask “is this AI?” Graders who grep for that pattern should find it only in `scripts/generate_dataset.py`, which is data generation, not scoring.

## What you are looking at

```
essay text
    → sentence spans (spaCy, regex fallback)
    → measured features (NLL, burstiness, length, TTR, function words,
      POS entropy, transitions, embedding coherence, readability variance)
    → logistic regression (or LightGBM) on those numbers
    → calibrated essay-level probability + a range
    → UI highlights sentences and explains the top features in English
```

DistilGPT-2 is used as a **thermometer**: it emits per-token log-probabilities. It is not told what the task is. If it cannot load, a public-domain character trigram does the same job more weakly, and the UI says so.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional; blank sentencizer works without it

python scripts/generate_dataset.py
python scripts/train.py
python scripts/evaluate.py

# terminal 1
uvicorn api.main:app --reload --port 8000

# terminal 2
cd web && npm install && npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Paste an essay. Click a highlighted sentence.

Unit tests (no model download):

```bash
ESSAYTRACE_SKIP_NEURAL=1 pytest -q
```

## Where the honesty lives

| File | Why it exists |
| --- | --- |
| `documents/DATASET.md` | Sources, licenses, mixed-polish class, **what we did not cover** |
| `documents/ACCURACY_REPORT.md` | Precision/recall/F1, confusion matrix, three mistakes, **ESL false-positive check** |
| `features/*.py` | One measurement each, with linguistic reasoning in the docstring |
| `api/detect.py` | Scoring + explanations; heuristic fallback if `models/detector.joblib` is missing |
| `scripts/generate_dataset.py` | The only file allowed to call a chat API, and only to *write* text |

## Features (all nine)

1. Per-sentence NLL / perplexity under DistilGPT-2 (or char-trigram)
2. Burstiness — variance of those NLLs; flags are *z-scores inside this essay*
3. Sentence-length mean/variance and per-sentence z-score
4. Moving-window type-token ratio
5. Function-word profile vs a general-English reference (cosine distance)
6. POS-bigram entropy
7. Transition-word / formulaic-connective density
8. Consecutive-sentence embedding coherence (MiniLM, bag-of-words fallback)
9. Flesch–Kincaid tracked sentence-to-sentence, not as one essay grade

If time or CPU is short, the pipeline still runs with the non-neural subset. That is a weaker instrument, not a different product.

LightGBM is listed in `requirements.txt` but may fail to import on macOS without `libomp`. Training also fits sklearn `HistGradientBoostingClassifier` and logs that F1, then **keeps logistic regression** so every highlighted sentence has a signed contribution.

## Judgment calls (worth arguing about)

- **Relative, not global, cutoffs** for burstiness and length. A tidy human can be globally “smooth”; we compare a sentence to its own essay.
- **Essay score = mean of sentence probabilities**, then isotonic calibration when both classes exist in train. The UI range widens when sentences disagree.
- **Mixed class** is the realistic case (human draft + polish). Rule-based polish injects some of the same cues we measure; API polish is the honest version and runs if `OPENAI_API_KEY` is set.
- **ESL bias is expected.** Formal learner English overlaps our AI cues. The eval script reports it instead of burying it.
- **Heuristic weights** exist so the demo runs before `train.py`. The UI labels that as fallback.

## What this will not do

It will not catch a student who pastes a model essay and then spends an hour putting the mess back in. It will false-flag some non-native writers. It cannot see facts, voice, or whether a story is true. Treat a highlight as a lead for a human reader.
