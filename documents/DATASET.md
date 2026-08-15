# Dataset card

This file is the graded artifact for how we built training and evaluation
data. Counts below are produced by `scripts/generate_dataset.py` (seed=14).

## The rule we followed

Live detection **never** sends essay text to a chat model. This generator
is the only place an API LLM may be called, and only to *write* synthetic
essays or a polish pass.

## Classes

| Class | What it means | Sentence labels |
| --- | --- | --- |
| `human` | Human-written throughout | every span `human` |
| `machine` | Model-written throughout | every span `ai` |
| `mixed` | Human draft with a subset of sentences polished | mixed `human` / `ai` |
| `esl` (eval only) | Learner-English probes, **not** used as AI labels | treated as `human` for the false-positive check |

## Sizes (this run)

- Human essays: **12**
- Machine essays: **8**
- Mixed / polished essays: **12**
- ESL eval probes: **8**
- Train / test split is by *essay id* (not by sentence) in `scripts/train.py`

### Sources actually used

- `seed_original+polish`: 12
- `seed_original`: 12
- `author_pastiche`: 8

### HC3 note

datasets library unavailable (No module named 'datasets').

### API generation / polish

No OPENAI_API_KEY (or the call failed). No second-model API samples this run. Mixed class used rule-based polish only.

## Human essays

**Primary (always present):** twelve original personal essays in
`scripts/seed_texts.py`, written for this repo. Register: first-person
admissions-like narrative (kitchens, contests, night shifts, translation).
License: original work in this repository.

**Secondary (if download succeeded):** human answers from
[Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
(CC-BY-SA). HC3 is Reddit/ELI5-style Q&A, not Common App essays. We
length-filter so passages are essay-sized, but **the domain is still
expository Q&A, not admissions narrative.** That is a coverage gap, not a
footnote.

We did **not** scrape copyrighted student Common App essays.

## Machine essays

**Preferred:** ChatGPT answers from the same HC3 dataset (CC-BY-SA), so
the machine class is *actual* instruction-tuned output, not our guess at
the style.

**Second model (if `OPENAI_API_KEY` is set):** new admissions essays
generated with the model in `ESSAYTRACE_GEN_MODEL` (default `gpt-4o-mini`).
That exists specifically so we do not overfit a single ChatGPT fingerprint.

**Fallback (this run WAS used (download/API produced too few machine essays)):** author-written pastiches
in a typical LLM register (Moreover / tapestry / "this experience taught
me"). These are **not** model samples. If the fallback fired, treat
machine-class metrics as optimistic for those surface cues and weaker for
true unseen models.

## Mixed provenance (the case most people skip)

Each mixed document starts from a human essay. A third to a half of
sentences are rewritten:

1. **Best:** an API "polish this, keep meaning" pass on selected sentences
   or the whole essay, then aligned back to spans when possible.
2. **Otherwise:** a rule-based polish (expand contractions, swap informal
   words, occasionally insert "Moreover,").

Unpolished sentences keep label `human`. Polished sentences get `ai`.
The detector therefore has to work **at sentence level**. A whole-essay
yes/no model would cheat on this class.

Limitation of the rule-based polish: it *injects the same surface features
we measure* (transitions, formality). That can inflate those features'
importance. API polish is the honest version of the task; rule-based
polish is a documented proxy.

## ESL / non-native check set

Eight constructed advanced-learner probes in `scripts/seed_texts.py`
(article errors, memorized formal chunks, flatter rhythm). They are
**not** official TOEFL items and **not** a substitute for a learner
corpus. They exist so `scripts/evaluate.py` can report a false-positive
rate instead of discovering ESL bias in a demo on stage.

If we later add a public learner corpus (e.g. W&I+LOCNESS), it should
replace these probes, not sit unnamed in the training pool.

## Topic coverage

Mostly: personal narrative and short expository answers — family work,
school clubs, moving, service, Q&A explainers.

## What this dataset does NOT cover

- STEM / heavily technical explanations, code-adjacent writing, research
  abstracts
- Creative fiction, poetry, drama
- Timed in-class handwriting transcribed with noise
- Non-English essays, or code-switching beyond short phrases
- Adversarial "humanizers" sold to evade detectors
- GPT-4 / Claude / Llama fingerprints unless the API path ran
- True professionally edited human essays (counselor edits ≠ LLM polish)
- Students who write in a very uniform formal register for other reasons
  (including some native speakers)

Performance claims outside that box are not supported by this file.

## License summary

| Piece | License |
| --- | --- |
| Seed human / ESL / pastiche text in this repo | original, bundled with the project |
| HC3 (if downloaded) | CC-BY-SA 4.0 via Hello-SimpleAI |
| API generations | check the provider ToS; we treat them as research samples, not for republishing as human work |

## Files written

- `data/processed/essays.jsonl` — one essay per line, with sentence annotations
- `data/processed/esl.jsonl` — ESL probes, eval only
- `data/processed/dataset_stats.json` — machine-readable copy of these counts
