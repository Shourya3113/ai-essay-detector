#!/usr/bin/env python3
"""Write documents/ACCURACY_REPORT.md from the held-out split.

Must include precision/recall/F1, a confusion matrix, three confident
mistakes with a theory, and an ESL false-positive comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.detect import detect  # noqa: E402
from features.extractor import FEATURE_NAMES, extract_essay, vectorize  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "documents"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def predict_bundle(bundle, vec: list[float]) -> float:
    import numpy as np

    x = np.array(vec, dtype=float).reshape(1, -1)
    scaler = bundle.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)
    clf = bundle["sentence_clf"]
    kind = bundle.get("kind", "logreg")
    if kind == "lightgbm":
        raw = float(np.ravel(clf.predict(x, raw_score=True))[0])
        p = 1.0 / (1.0 + np.exp(-raw))
    elif hasattr(clf, "predict_proba"):
        p = float(clf.predict_proba(x)[0, 1])
    else:
        p = float(clf.decision_function(x)[0])
        p = 1.0 / (1.0 + np.exp(-p))
    return p


def fmt_matrix(cm: np.ndarray) -> str:
    return (
        "| | Pred human | Pred AI |\n"
        "| --- | ---: | ---: |\n"
        f"| **True human** | {cm[0, 0]} | {cm[0, 1]} |\n"
        f"| **True AI** | {cm[1, 0]} | {cm[1, 1]} |\n"
    )


def top_feats(features: dict[str, float], importances: dict[str, float], k: int = 5) -> str:
    scored = []
    for name in FEATURE_NAMES:
        scored.append((name, features.get(name, 0.0), importances.get(name, 0.0)))
    scored.sort(key=lambda t: abs(t[2] * t[1]), reverse=True)
    lines = []
    for name, val, imp in scored[:k]:
        lines.append(f"  - `{name}` = {val:.3f} (model weight {imp:+.3f})")
    return "\n".join(lines)


def theory_for(row: dict, pred_ai: bool, features: dict, *, correct: bool) -> str:
    if correct:
        return (
            "Not an error — the model was confidently right on this one. On a "
            "tiny split whose machine class is author-written pastiche, that is "
            "cheap confidence. It does not mean the same essay would be scored "
            "honestly against GPT-4 or a human who writes in a formal L2 register."
        )
    label = row.get("essay_label")
    if pred_ai and label in {"human", "esl_human"}:
        if row.get("esl") or label == "esl_human":
            return (
                "Theory: learner English often uses memorized formal chunks and "
                "flatter rhythm, which overlap our AI cues (transition density, "
                "low length variance, even readability). This is the classic ESL "
                "false-positive mode — not 'the student cheated.'"
            )
        if features.get("essay_transition_density", 0) > 0.02:
            return (
                "Theory: this human draft uses explicit discourse markers or a "
                "very regular sentence engine, so glue-word and uniformity cues "
                "fired. Formal native writing can look like polish."
            )
        if features.get("essay_burstiness", 1) < 0.2:
            return (
                "Theory: surprisal was unusually flat. A careful, heavily revised "
                "human paragraph can be as even as a model sample."
            )
        return (
            "Theory: within-essay uniformity (length, readability, or neighbor "
            "embedding similarity) outweighed the messy human details. The "
            "classifier has no access to fact-checking or voice — only measured "
            "shape — so a tidy human can look generated."
        )
    # missed AI
    if features.get("essay_transition_density", 1) < 0.01:
        return (
            "Theory: this machine (or polished) text did not overuse the obvious "
            "connectives, so the highest-precision lexical cue was absent. "
            "Without DistilGPT-2 loaded, surprisal also collapses toward a weak "
            "character model and the sample can hide."
        )
    return (
        "Theory: mixed provenance diluted the essay-level cues — leftover human "
        "sentences pulled the mean toward 'human' even where a polished span "
        "was locally smooth. Span-level recall is the harder problem, and we "
        "missed this one."
    )


def main() -> None:
    bundle_path = MODELS / "detector.joblib"
    split_path = PROCESSED / "split.json"
    essays_path = PROCESSED / "essays.jsonl"
    esl_path = PROCESSED / "esl.jsonl"
    if not bundle_path.exists() or not essays_path.exists():
        raise SystemExit("Need models/detector.joblib and data/processed/essays.jsonl")

    bundle = joblib.load(bundle_path)
    essays = {r["id"]: r for r in load_jsonl(essays_path)}
    if split_path.exists():
        test_ids = set(json.loads(split_path.read_text())["test_essay_ids"])
    else:
        test_ids = set(list(essays)[::4])

    test_rows = [essays[i] for i in test_ids if i in essays]
    importances = bundle.get("importances") or {}

    y_true = []
    y_pred = []
    y_prob = []
    per_essay = []
    for row in test_rows:
        rec = extract_essay(row["text"])
        if not rec.sentences:
            continue
        ps = [predict_bundle(bundle, vectorize(s)) for s in rec.sentences]
        p = float(np.mean(ps))
        cal = bundle.get("essay_calibrator")
        if cal is not None:
            try:
                p = float(cal.predict([p])[0])
            except Exception:
                pass
        gold = 1 if row.get("essay_label") in {"machine", "mixed"} else 0
        pred = 1 if p >= 0.5 else 0
        y_true.append(gold)
        y_pred.append(pred)
        y_prob.append(p)
        per_essay.append((row, rec, p, gold, pred))

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    if len(np.unique(y_true_a)) < 2:
        # Degenerate split — still report what we can.
        prec = reca = f1 = float("nan")
        cm = np.array([[0, 0], [0, 0]])
    else:
        prec = precision_score(y_true_a, y_pred_a, zero_division=0)
        reca = recall_score(y_true_a, y_pred_a, zero_division=0)
        f1 = f1_score(y_true_a, y_pred_a, zero_division=0)
        cm = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1])

    report = classification_report(
        y_true_a, y_pred_a, zero_division=0, target_names=["human", "ai"]
    ) if len(y_true_a) else "no test rows"

    # Confident mistakes: wrong, sorted by |p-0.5|
    mistakes = [t for t in per_essay if t[3] != t[4]]
    mistakes.sort(key=lambda t: abs(t[2] - 0.5), reverse=True)
    near_misses = sorted(per_essay, key=lambda t: abs(t[2] - 0.5))
    chosen = mistakes[:3]
    used_near = False
    if len(chosen) < 3:
        used_near = True
        have = {id(t[0]) for t in chosen}
        for t in near_misses:
            if id(t[0]) in have:
                continue
            chosen.append(t)
            if len(chosen) == 3:
                break

    mistake_md = []
    for i, (row, rec, p, gold, pred) in enumerate(chosen, 1):
        wrong = gold != pred
        heading = "confident error" if wrong else "nearest miss (not an error — not enough test mistakes)"
        gold_s = "AI/mixed" if gold else "human"
        pred_s = "AI" if pred else "human"
        snippet = row["text"][:280].replace("\n", " ")
        feats = rec.essay_features
        mistake_md.append(
            f"### Example {i} — {heading}\n\n"
            f"- id: `{row['id']}`\n"
            f"- source: `{row.get('source')}`\n"
            f"- gold essay label: **{gold_s}**\n"
            f"- detector point estimate: **{p:.2f}** → predicted **{pred_s}**\n"
            f"- excerpt: {snippet}…\n\n"
            f"Feature values (essay-level plus a few span cues):\n"
            f"{top_feats(feats, importances)}\n\n"
            f"{theory_for(row, pred == 1, feats, correct=not wrong)}\n"
        )

    # ESL check via the same detect() the app uses, so we don't cheat.
    esl_rows = load_jsonl(esl_path) if esl_path.exists() else []
    esl_scores = []
    for row in esl_rows:
        out = detect(row["text"])
        esl_scores.append(out.essay_score.point)
    esl_fp = sum(1 for p in esl_scores if p >= 0.5)
    esl_rate = (esl_fp / len(esl_scores)) if esl_scores else float("nan")

    human_test = [t for t in per_essay if t[3] == 0]
    human_fp = sum(1 for t in human_test if t[4] == 1)
    human_rate = (human_fp / len(human_test)) if human_test else float("nan")

    if esl_rate == esl_rate and human_rate == human_rate:
        if esl_rate > human_rate + 0.05:
            esl_verdict = (
                f"Yes — ESL probes flagged at **{esl_rate:.0%}** vs **{human_rate:.0%}** "
                f"false positives on held-out human essays. We are treating that as a "
                f"real failure mode, not a rounding error."
            )
        elif esl_rate > human_rate:
            esl_verdict = (
                f"Slightly elevated: ESL **{esl_rate:.0%}** vs other humans **{human_rate:.0%}**. "
                f"The sample is small; direction matches the known bias."
            )
        else:
            esl_verdict = (
                f"Not elevated on this tiny probe set (ESL **{esl_rate:.0%}** vs humans "
                f"**{human_rate:.0%}**). We do **not** claim the detector is ESL-safe — "
                f"the probes are constructed and N={len(esl_scores)}."
            )
    else:
        esl_verdict = "Could not compute a comparison (empty split or empty ESL file)."

    neural = detect("This is a short calibration sentence. It exists so we can ask the API which instruments loaded. Another sentence follows for length.").instruments

    if len(y_true_a) < 20 or (f1 == f1 and f1 >= 0.99):
        honesty = (
            f"**Read this first:** essay-level F1 is {f1:.3f} on **N={len(y_true_a)}** "
            f"held-out essays. The machine class this run is mostly author-written "
            f"LLM-register pastiche plus rule-based polish, and the top weights are "
            f"transition-word counts — features we *injected* into the fallback "
            f"machine/mixed data. This is not a claim that the detector works on "
            f"unseen ChatGPT/Claude essays. Sentence-level F1 from training "
            f"({bundle.get('metrics', {}).get('sentence_f1_logreg')}) is the less "
            f"inflated number. The ESL table is the one to remember."
        )
    else:
        honesty = (
            "Essay-level F1 below is still a small-N estimate. Pair it with the "
            "confusion matrix and the ESL false-positive row."
        )

    md = f"""# Accuracy report

Auto-generated by `scripts/evaluate.py`. Numbers are from a **held-out
essay-level split**, not from training sentences. If a cell looks too
round or too perfect, assume overfitting and read the caveats.

{honesty}

## Headline metrics (essay-level, threshold 0.5)

These are **not** a single accuracy percentage. Precision/recall/F1 on
the held-out essays (label `machine` or `mixed` = positive):

| Metric | Value |
| --- | ---: |
| Precision | {prec:.3f} |
| Recall | {reca:.3f} |
| F1 | {f1:.3f} |
| N essays | {len(y_true_a)} |
| Model | `{bundle.get("model_name")}` |

Confusion matrix:

{fmt_matrix(cm)}

Sentence-level sklearn report (same split, mean-of-sentences thresholded
only at essay level above; this block is the essay classification report):

```
{report}
```

Training-time notes from the bundle:

```
{json.dumps(bundle.get("metrics", {}), indent=2)}
```

## Instruments actually used at eval time

```
{json.dumps({k: neural[k] for k in neural if k != "feature_names"}, indent=2)}
```

If `gpt2` is false, DistilGPT-2 did not load and NLL came from the
character trigram fallback — weaker measurement, more honest than
pretending we had a transformer.

## Three examples the detector gets (or nearly gets) wrong

{chr(10).join(mistake_md)}
{"We did not have three held-out mistakes, so we filled with nearest misses. That usually means the test split is tiny or the labels are too easy (especially if machine class is pastiche). Do not read a short mistake list as 'the detector works.'" if used_near else ""}

## ESL bias check

The brief asks whether we noticed the non-native English false-positive
problem. We ran the **same** `detect()` path used by the web app on
`data/processed/esl.jsonl` (constructed advanced-learner probes; see
DATASET.md).

| Set | N | Flagged AI at 0.5 | Rate |
| --- | ---: | ---: | ---: |
| ESL probes | {len(esl_scores)} | {esl_fp} | {esl_rate if esl_rate == esl_rate else float("nan"):.3f} |
| Held-out human essays | {len(human_test)} | {human_fp} | {human_rate if human_rate == human_rate else float("nan"):.3f} |

**Verdict:** {esl_verdict}

Why this happens even when we are careful: several of our strongest
*measurements* (even sentence length, discourse markers from classroom
English, lower burstiness, high neighbor coherence after a student has
been taught "use linking words") are also features of L2 academic
writing. A detector that cannot see identity will punish some ESL
students. That is not a side note. **Do not use this tool as an
integrity verdict on international applicants.**

## Calibration and the range the UI shows

Essay score = mean of sentence probabilities, passed through isotonic
regression when the training split had both classes. The UI range is
**not** a confidence interval from a Bayesian model. It is the point
estimate plus a half-width that grows with sentence-to-sentence
disagreement. Mixed essays *should* produce wide ranges.

## What would make these numbers go up without making the product more honest

- Training on the same LLM you then "detect"
- Reporting token-level accuracy on sentences leaked from the same essay
- Quoting training accuracy
- Hiding the ESL table

We are not doing those.

## Feature importances (signed for logistic regression; split gain for LightGBM)

"""
    ranked = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    md += "| Feature | Weight |\n| --- | ---: |\n"
    for name, val in ranked:
        md += f"| `{name}` | {val:+.4f} |\n"

    md += """
Positive log-reg weights push toward the AI class after scaling. Read
them as "what the linear model used," not as causal linguistics.

## Bottom line

Treat every flagged sentence as a **lead for a human reader**, not as
evidence that would survive an honor-code hearing. The product is built
to show *where* and *why* a measurement looked unusual. That is the
ceiling of what this architecture can honestly claim.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ACCURACY_REPORT.md").write_text(md, encoding="utf-8")
    print("Wrote", DOCS / "ACCURACY_REPORT.md")
    print(f"F1={f1:.3f}  ESL_FP={esl_rate}  human_FP={human_rate}")


if __name__ == "__main__":
    main()
