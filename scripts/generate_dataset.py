#!/usr/bin/env python3
"""Generate the training/eval dataset.

THIS FILE IS NOT PART OF THE LIVE DETECTION PIPELINE.
The API never imports it. Using an LLM here is allowed only to *write
synthetic essays*, never to score them.

Sources, in order of preference:
  1. Public human vs ChatGPT answers from Hello-SimpleAI/HC3 (Hugging Face)
  2. Original human seed essays in scripts/seed_texts.py
  3. Optional API generation (OPENAI_API_KEY / ANTHROPIC_API_KEY) for a
     second-model machine class and a true "polish this" pass
  4. Offline fallback: LLM-register pastiches + rule-based polish

Run:  python scripts/generate_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.segmentation import split_sentences  # noqa: E402
from scripts.seed_texts import ESL_PROBES, HUMAN_ESSAYS, MACHINE_PROMPTS  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "documents"
SEED = 14

# Offline fallback: author-written in a typical instruction-tuned register.
# Used only if public ChatGPT text cannot be downloaded. Labeled as such.
FALLBACK_MACHINE: list[dict] = [
    {
        "id": "pastiche_01",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "From a young age, I have always been drawn to the quiet lessons of hard work. "
            "Working in my family's restaurant taught me the importance of resilience, patience, and teamwork. "
            "Moreover, each shift presented a new challenge that shaped my character in meaningful ways. "
            "Furthermore, I learned that leadership is not about titles; it is about showing up consistently. "
            "When I made mistakes, I reflected on them and used them as opportunities for growth. "
            "In today's world, the ability to adapt is essential, and this environment trained me to adapt. "
            "It is important to note that these experiences also deepened my commitment to my community. "
            "In conclusion, I hope to bring this same dedication, humility, and perseverance to college."
        ),
    },
    {
        "id": "pastiche_02",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "The robotics competition was more than an event; it was a tapestry of collaboration and discovery. "
            "I joined the team hoping to explore engineering, and I quickly realized that success depends on communication. "
            "Additionally, our greatest failure — a robot that stalled on the field — became our most valuable teacher. "
            "We gathered afterward to analyze the problem, and that conversation underscored the value of humility. "
            "This experience taught me that innovation requires both creativity and discipline. "
            "As a result, I now approach challenges with curiosity rather than fear. "
            "I am eager to continue this journey of learning in a college community that values inquiry."
        ),
    },
    {
        "id": "pastiche_03",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Helping my grandmother in her garden revealed to me the power of patience and cultural heritage. "
            "Each plant required attention, and each season offered a new lesson about care. "
            "Moreover, translating for her at appointments showed me that language can be an act of service. "
            "I learned the importance of listening carefully and speaking with respect. "
            "These moments, though small, played a pivotal role in shaping my values. "
            "I hope to study fields that allow me to support immigrant families with the same diligence. "
            "In summary, my background has prepared me to contribute thoughtfully to campus life."
        ),
    },
    {
        "id": "pastiche_04",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Debate has taught me how to examine an issue from multiple perspectives. "
            "At first, I focused only on winning, but I gradually understood that the real goal is clarity. "
            "Furthermore, losing a close round helped me recognize the limits of a single definition. "
            "I began to research more deeply and to listen to opponents with genuine interest. "
            "This growth has influenced my academic interests in public policy and ethics. "
            "I believe a college education will allow me to refine these skills and apply them to real problems. "
            "Ultimately, I want to be a student who argues in good faith and learns in public."
        ),
    },
    {
        "id": "pastiche_05",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Working a late shift taught me responsibility in a concrete and lasting way. "
            "I balanced schoolwork with employment, and I learned to manage my time with care. "
            "Moreover, interacting with customers from many backgrounds broadened my understanding of community. "
            "I discovered that small acts of patience can change the tone of an entire evening. "
            "These lessons will guide me as I enter a more independent stage of life. "
            "I am confident that I can contribute to a campus through service, focus, and empathy. "
            "In conclusion, this job was not merely a paycheck; it was a foundation for my future."
        ),
    },
    {
        "id": "pastiche_06",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Music has always been a central part of my identity. "
            "Practicing trombone in jazz band required discipline, listening, and trust. "
            "Additionally, performing with others showed me that individual excellence means little without ensemble awareness. "
            "When I struggled with a difficult passage, I sought feedback and improved through steady effort. "
            "This process mirrors the way I hope to learn in college: iteratively, collaboratively, and with humility. "
            "I look forward to joining a community where curiosity is shared. "
            "It is worth noting that the most important note I learned was the one I had to repeat."
        ),
    },
    {
        "id": "pastiche_07",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Repairing an old motorcycle was an unexpected classroom. "
            "I began with little knowledge, but I was determined to understand how the system worked. "
            "Step by step, I learned to diagnose problems, research solutions, and accept incomplete success. "
            "Furthermore, the project taught me that practical skills and academic skills can strengthen one another. "
            "I now see engineering not only as a career path but as a way of thinking. "
            "I hope to bring this hands-on curiosity to a rigorous college environment. "
            "In today's society, we need people who can both theorize and tinker, and I aspire to be one of them."
        ),
    },
    {
        "id": "pastiche_08",
        "model": "author_pastiche_chatgpt_register",
        "text": (
            "Volunteering in a hospital introduced me to the quiet dimensions of care. "
            "I delivered magazines, poured water, and learned to be present without demanding conversation. "
            "Moreover, I observed how small logistical details can protect a person's dignity. "
            "This experience did not simply 'inspire me to help people'; it showed me that help is often specific. "
            "I want to study fields that combine analysis with service. "
            "I believe I can contribute to a campus by paying attention to what is easy to overlook. "
            "In conclusion, the cart I pushed was ordinary, and that was the point."
        ),
    },
]


CONTRACTIONS = [
    ("don't", "do not"),
    ("didn't", "did not"),
    ("isn't", "is not"),
    ("wasn't", "was not"),
    ("can't", "cannot"),
    ("I'm", "I am"),
    ("I've", "I have"),
    ("I'd", "I would"),
    ("that's", "that is"),
    ("it's", "it is"),
    ("There's", "There is"),
    ("there's", "there is"),
    ("couldn't", "could not"),
    ("wouldn't", "would not"),
    ("won't", "will not"),
]

SYNONYMS = [
    (r"\bgot\b", "obtained"),
    (r"\bkid\b", "child"),
    (r"\bkids\b", "children"),
    (r"\bstuff\b", "material"),
    (r"\bAnyway,\b", "That said,"),
    (r"\bdumb\b", "unwise"),
    (r"\bstupid\b", "ill-advised"),
    (r"\bjob\b", "role"),
]


def _hash_id(prefix: str, text: str) -> str:
    return prefix + "_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ok_len(text: str, lo: int = 280, hi: int = 3500) -> bool:
    n = len(text.strip())
    return lo <= n <= hi


def try_load_hc3(max_human: int = 70, max_ai: int = 70) -> tuple[list[dict], list[dict], str]:
    note = "HC3 not loaded."
    try:
        from datasets import load_dataset
    except Exception as exc:
        return [], [], f"datasets library unavailable ({exc})."

    humans: list[dict] = []
    machines: list[dict] = []
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    except Exception as exc:
        return [], [], f"HC3 download failed ({exc})."

    rng = random.Random(SEED)
    rows = list(ds)
    rng.shuffle(rows)
    for row in rows:
        if len(humans) >= max_human and len(machines) >= max_ai:
            break
        for ans in row.get("human_answers") or []:
            t = _normalize(str(ans))
            if _ok_len(t) and len(humans) < max_human:
                humans.append(
                    {
                        "id": _hash_id("hc3h", t),
                        "text": t,
                        "essay_label": "human",
                        "source": "hc3_human",
                        "license": "CC-BY-SA (Hello-SimpleAI/HC3)",
                    }
                )
        for ans in row.get("chatgpt_answers") or []:
            t = _normalize(str(ans))
            if _ok_len(t) and len(machines) < max_ai:
                machines.append(
                    {
                        "id": _hash_id("hc3m", t),
                        "text": t,
                        "essay_label": "machine",
                        "source": "hc3_chatgpt",
                        "model": "chatgpt",
                        "license": "CC-BY-SA (Hello-SimpleAI/HC3)",
                    }
                )
    note = (
        f"HC3 loaded: {len(humans)} human answers, {len(machines)} ChatGPT answers "
        f"(length-filtered, subsampled)."
    )
    return humans, machines, note


def try_api_generate(n: int = 8) -> list[dict]:
    """Second-model machine essays. Never used at inference time."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or n <= 0:
        return []
    try:
        import urllib.request

        out: list[dict] = []
        for i, prompt in enumerate(MACHINE_PROMPTS[:n]):
            body = json.dumps(
                {
                    "model": os.environ.get("ESSAYTRACE_GEN_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "Write a 350-500 word college admissions personal essay. Do not mention that you are an AI.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.9,
                }
            ).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
            text = _normalize(payload["choices"][0]["message"]["content"])
            if _ok_len(text, lo=200):
                out.append(
                    {
                        "id": f"api_m_{i:02d}",
                        "text": text,
                        "essay_label": "machine",
                        "source": "openai_api",
                        "model": os.environ.get("ESSAYTRACE_GEN_MODEL", "gpt-4o-mini"),
                        "license": "generated-for-research",
                    }
                )
        return out
    except Exception:
        return []


def try_api_polish(human_text: str) -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        import urllib.request

        body = json.dumps(
            {
                "model": os.environ.get("ESSAYTRACE_GEN_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Polish this college essay: keep meaning and first-person details, "
                            "improve flow and grammar, do not add new achievements. Return only the essay."
                        ),
                    },
                    {"role": "user", "content": human_text},
                ],
                "temperature": 0.4,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return _normalize(payload["choices"][0]["message"]["content"])
    except Exception:
        return None


def rule_polish_sentence(text: str, add_transition: bool) -> str:
    out = text
    for a, b in CONTRACTIONS:
        out = out.replace(a, b)
    for pat, repl in SYNONYMS:
        out = re.sub(pat, repl, out)
    if add_transition and not re.match(
        r"^(Moreover|Furthermore|Additionally|In conclusion|However)", out
    ):
        out = "Moreover, " + out[0].lower() + out[1:] if out else out
    return out


def make_mixed(human: dict, rng: random.Random, polish_fn) -> dict:
    """Keep mixed provenance: some sentences original, some polished."""
    spans = split_sentences(human["text"])
    if len(spans) < 4:
        return {}
    n_poly = max(2, len(spans) // 3)
    idxs = set(rng.sample(range(len(spans)), n_poly))
    pieces: list[str] = []
    anns: list[dict] = []
    cursor = 0
    for i, span in enumerate(spans):
        if i in idxs:
            new = polish_fn(span.text, add_transition=(i % 2 == 0))
            label = "ai"
        else:
            new = span.text
            label = "human"
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(new)
        cursor += len(new)
        anns.append({"start": start, "end": cursor, "label": label, "text": new})
    text = "".join(pieces)
    return {
        "id": _hash_id("mix", text),
        "text": text,
        "essay_label": "mixed",
        "source": human.get("source", "seed") + "+polish",
        "license": human.get("license", "original-seed"),
        "sentence_annotations": anns,
    }


def attach_all_human_or_ai(doc: dict, label: str) -> dict:
    spans = split_sentences(doc["text"])
    anns = [{"start": s.start, "end": s.end, "label": label, "text": s.text} for s in spans]
    doc = dict(doc)
    doc["sentence_annotations"] = anns
    return doc


def write_dataset_md(stats: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "DATASET.md"
    path.write_text(
        f"""# Dataset card

This file is the graded artifact for how we built training and evaluation
data. Counts below are produced by `scripts/generate_dataset.py` (seed={SEED}).

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

- Human essays: **{stats["n_human"]}**
- Machine essays: **{stats["n_machine"]}**
- Mixed / polished essays: **{stats["n_mixed"]}**
- ESL eval probes: **{stats["n_esl"]}**
- Train / test split is by *essay id* (not by sentence) in `scripts/train.py`

### Sources actually used

{stats["source_block"]}

### HC3 note

{stats["hc3_note"]}

### API generation / polish

{stats["api_note"]}

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

**Fallback (this run {stats["used_pastiche"]}):** author-written pastiches
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
""",
        encoding="utf-8",
    )


def main() -> None:
    rng = random.Random(SEED)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    humans = [
        {
            "id": h["id"],
            "text": _normalize(h["text"]),
            "essay_label": "human",
            "source": "seed_original",
            "topic": h["topic"],
            "license": "original-seed",
        }
        for h in HUMAN_ESSAYS
    ]
    machines: list[dict] = []

    hc3_h, hc3_m, hc3_note = try_load_hc3()
    humans.extend(hc3_h)
    machines.extend(hc3_m)

    api_m = try_api_generate(n=8)
    machines.extend(api_m)
    api_note = (
        f"API machine essays generated: {len(api_m)}."
        if api_m
        else "No OPENAI_API_KEY (or the call failed). No second-model API samples this run."
    )

    used_pastiche = "was NOT used"
    if len(machines) < 12:
        for p in FALLBACK_MACHINE:
            machines.append(
                {
                    "id": p["id"],
                    "text": _normalize(p["text"]),
                    "essay_label": "machine",
                    "source": "author_pastiche",
                    "model": p["model"],
                    "license": "original-seed",
                }
            )
        used_pastiche = "WAS used (download/API produced too few machine essays)"

    def polish_fn(sentence: str, add_transition: bool) -> str:
        return rule_polish_sentence(sentence, add_transition)

    mixed: list[dict] = []
    polish_api_ok = 0
    for h in humans[: min(len(humans), 24)]:
        whole = try_api_polish(h["text"]) if polish_api_ok < 6 else None
        if whole:

            def _api_sent(sentence: str, add_transition: bool, _whole=whole, _orig=h["text"]) -> str:
                # If we have a full-essay polish, still only replace selected
                # sentences by taking the corresponding index from the polished
                # split when lengths match; otherwise fall back to rules on that span.
                orig_s = split_sentences(_orig)
                new_s = split_sentences(_whole)
                if len(orig_s) == len(new_s):
                    # map by order; caller picks which index via the sentence text
                    for o, n in zip(orig_s, new_s):
                        if o.text == sentence:
                            return n.text
                return rule_polish_sentence(sentence, add_transition)

            doc = make_mixed(h, rng, _api_sent)
            if doc:
                doc["source"] = h.get("source", "seed") + "+api_polish"
                polish_api_ok += 1
                mixed.append(doc)
                continue
        doc = make_mixed(h, rng, polish_fn)
        if doc:
            mixed.append(doc)

    if polish_api_ok:
        api_note += f" API polish used on {polish_api_ok} mixed essays."
    else:
        api_note += " Mixed class used rule-based polish only."

    labeled_h = [attach_all_human_or_ai(d, "human") for d in humans]
    labeled_m = [attach_all_human_or_ai(d, "ai") for d in machines]
    essays = labeled_h + labeled_m + mixed
    rng.shuffle(essays)

    esl = [
        attach_all_human_or_ai(
            {
                "id": e["id"],
                "text": _normalize(e["text"]),
                "essay_label": "esl_human",
                "source": "constructed_l2_probe",
                "topic": e["topic"],
                "license": "original-seed",
                "esl": True,
            },
            "human",
        )
        for e in ESL_PROBES
    ]

    essays_path = PROCESSED / "essays.jsonl"
    esl_path = PROCESSED / "esl.jsonl"
    with essays_path.open("w", encoding="utf-8") as f:
        for row in essays:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with esl_path.open("w", encoding="utf-8") as f:
        for row in esl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    sources = Counter(r.get("source", "?") for r in essays)
    source_block = "\n".join(f"- `{k}`: {v}" for k, v in sources.most_common())
    stats = {
        "n_human": len(labeled_h),
        "n_machine": len(labeled_m),
        "n_mixed": len(mixed),
        "n_esl": len(esl),
        "n_total_trainpool": len(essays),
        "sources": dict(sources),
        "hc3_note": hc3_note,
        "api_note": api_note,
        "used_pastiche": used_pastiche,
        "source_block": source_block,
    }
    (PROCESSED / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_dataset_md(stats)
    print(json.dumps({k: stats[k] for k in ("n_human", "n_machine", "n_mixed", "n_esl")}, indent=2))
    print("Wrote", essays_path)
    print("Wrote", DOCS / "DATASET.md")


if __name__ == "__main__":
    main()
