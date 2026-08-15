"""Turn measured features into a score + evidence.

The classifier is logistic regression or LightGBM trained on *features*,
never on a chat model's opinion. If no trained artifact is on disk we
fall back to a documented heuristic weighted sum so the app still runs,
and we tell the UI that happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from api.schemas import DetectResponse, EssayScore, FeatureHit, SentenceResult
from features.extractor import FEATURE_NAMES, extract_essay, vectorize
from features.nlp_resources import skip_neural

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "detector.joblib"

CAVEAT = (
    "No detector is perfect — treat this as one signal, not a verdict. "
    "Scores are calibrated probabilities from measured linguistic features, "
    "with a range that reflects how much sentences in this essay disagree. "
    "Non-native English writing is a known false-positive risk."
)

# Used only when models/detector.joblib is missing. Signs follow the
# usual research findings (smoother, more uniform, more discourse glue
# → higher P(AI)). These weights are a judgment call, not fitted.
HEURISTIC_WEIGHTS: dict[str, float] = {
    "nll_z": -0.35,
    "word_count_z": -0.08,
    "window_ttr": -0.25,
    "transition_count": 0.55,
    "transition_density": 0.7,
    "embed_sim_prev": 0.4,
    "embed_sim_next": 0.25,
    "fk_z": -0.05,
    "pos_local_entropy": -0.2,
    "essay_burstiness": -0.45,
    "essay_len_var": -0.3,
    "essay_ttr": -0.2,
    "essay_fw_cosine": 0.15,
    "essay_pos_entropy": -0.25,
    "essay_transition_density": 0.5,
    "essay_embed_coherence_mean": 0.45,
    "essay_fk_var": -0.3,
}

PLAIN_PRIORITY = (
    "nll_z",
    "essay_burstiness",
    "transition_density",
    "transition_count",
    "essay_transition_density",
    "embed_sim_prev",
    "essay_embed_coherence_mean",
    "word_count_z",
    "essay_len_var",
    "window_ttr",
    "essay_ttr",
    "pos_local_entropy",
    "essay_pos_entropy",
    "fk_z",
    "essay_fk_var",
    "essay_fw_cosine",
)


_BUNDLE: dict[str, Any] | None = None


def load_bundle() -> dict[str, Any] | None:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    if not MODEL_PATH.exists():
        return None
    import joblib

    loaded = joblib.load(MODEL_PATH)
    if not isinstance(loaded, dict) or "sentence_clf" not in loaded:
        return None
    _BUNDLE = loaded
    return _BUNDLE


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _heuristic_score(vec: list[float]) -> tuple[float, dict[str, float]]:
    contrib: dict[str, float] = {}
    total = -0.4
    for name, value in zip(FEATURE_NAMES, vec):
        w = HEURISTIC_WEIGHTS.get(name, 0.0)
        c = w * value
        contrib[name] = c
        total += c
    return _sigmoid(total), contrib


def _model_score(vec: list[float], bundle: dict[str, Any]) -> tuple[float, dict[str, float]]:
    x = np.array(vec, dtype=float).reshape(1, -1)
    scaler = bundle.get("scaler")
    if scaler is not None:
        x_scaled = scaler.transform(x)
    else:
        x_scaled = x

    clf = bundle["sentence_clf"]
    kind = bundle.get("kind", "logreg")
    contrib: dict[str, float] = {n: 0.0 for n in FEATURE_NAMES}

    if kind == "lightgbm":
        booster = clf.booster_ if hasattr(clf, "booster_") else clf
        raw = booster.predict(x_scaled, raw_score=True)
        p = _sigmoid(float(np.ravel(raw)[0]))
        try:
            shap = booster.predict(x_scaled, pred_contrib=True)
            row = np.ravel(shap)[:-1]
            for name, c in zip(FEATURE_NAMES, row):
                contrib[name] = float(c)
        except Exception:
            pass
    else:
        if hasattr(clf, "predict_proba"):
            p = float(clf.predict_proba(x_scaled)[0, 1])
        else:
            p = _sigmoid(float(np.ravel(clf.decision_function(x_scaled))[0]))
        inner = clf
        if hasattr(clf, "calibrated_classifiers_"):
            c0 = clf.calibrated_classifiers_[0]
            inner = getattr(c0, "estimator", getattr(c0, "base_estimator", clf))
        if hasattr(inner, "coef_"):
            parts = np.ravel(inner.coef_) * x_scaled.ravel()
            for name, c in zip(FEATURE_NAMES, parts):
                contrib[name] = float(c)

    calibrator = bundle.get("sentence_calibrator")
    if calibrator is not None:
        try:
            p = float(calibrator.predict([p])[0])
        except Exception:
            try:
                p = float(calibrator.predict_proba(np.array([[p]]))[0, 1])
            except Exception:
                pass
    return max(0.0, min(1.0, p)), contrib


def _top_explanations(
    contrib: dict[str, float],
    explanations: dict[str, str],
    features: dict[str, float],
    k: int = 3,
) -> list[FeatureHit]:
    # Features that pushed toward the AI class (positive contribution).
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    hits: list[FeatureHit] = []
    seen: set[str] = set()
    for name, c in ranked:
        if c <= 0:
            continue
        plain = explanations.get(name) or explanations.get(_family(name))
        if not plain:
            continue
        hits.append(
            FeatureHit(
                feature=name,
                value=float(features.get(name, 0.0)),
                contribution=float(c),
                plain=plain,
            )
        )
        seen.add(name)
        if len(hits) >= k:
            return hits
    # Fill from priority list so the panel is never empty.
    for name in PLAIN_PRIORITY:
        if name in seen:
            continue
        plain = explanations.get(name)
        if not plain:
            continue
        hits.append(
            FeatureHit(
                feature=name,
                value=float(features.get(name, 0.0)),
                contribution=float(contrib.get(name, 0.0)),
                plain=plain,
            )
        )
        if len(hits) >= k:
            break
    return hits[:k]


def _family(name: str) -> str:
    if name.startswith("essay_"):
        return name
    return name


def _aggregate(probs: list[float], bundle: dict[str, Any] | None) -> tuple[float, float, float]:
    if not probs:
        return 0.5, 0.25, 0.75
    arr = np.array(probs, dtype=float)
    point = float(np.mean(arr))
    if bundle and bundle.get("essay_calibrator") is not None:
        cal = bundle["essay_calibrator"]
        try:
            point = float(cal.predict([point])[0])
        except Exception:
            try:
                point = float(cal.predict_proba(np.array([[point]]))[0, 1])
            except Exception:
                pass
    point = max(0.02, min(0.98, point))
    spread = float(np.std(arr))
    half = min(0.28, max(0.08, 0.08 + 0.35 * spread))
    low = max(0.0, point - half)
    high = min(1.0, point + half)
    return point, low, high


def _label(point: float) -> str:
    if point < 0.35:
        return "More consistent with human drafting"
    if point < 0.55:
        return "Mixed / uncertain signal"
    if point < 0.75:
        return "Elevated AI-writing signal"
    return "Strong AI-writing signal"


def detect(text: str) -> DetectResponse:
    bundle = load_bundle()
    use_neural = not skip_neural()
    if bundle is not None and "skip_neural" in bundle:
        use_neural = not bool(bundle["skip_neural"])
    stripped = text.strip()
    record = extract_essay(stripped, use_neural=use_neural)
    model_name = "heuristic-fallback"
    if bundle:
        model_name = str(bundle.get("model_name", bundle.get("kind", "trained")))

    sentences_out: list[SentenceResult] = []
    probs: list[float] = []
    for sent in record.sentences:
        vec = vectorize(sent)
        if bundle:
            p, contrib = _model_score(vec, bundle)
        else:
            p, contrib = _heuristic_score(vec)
        probs.append(p)
        sentences_out.append(
            SentenceResult(
                text=sent.span.text,
                start=sent.span.start,
                end=sent.span.end,
                p_ai=p,
                explanations=_top_explanations(contrib, sent.explanations, sent.features),
            )
        )

    too_short = len(record.sentences) < 3
    if too_short:
        point, low, high = 0.5, 0.2, 0.8
        label = "Too short to score honestly"
    else:
        point, low, high = _aggregate(probs, bundle)
        label = _label(point)

    instruments = {
        **record.neural_used,
        "trained_classifier": bundle is not None,
        "feature_names": list(FEATURE_NAMES),
    }
    return DetectResponse(
        essay_score=EssayScore(point=point, low=low, high=high, label=label),
        caveat=CAVEAT,
        sentences=sentences_out,
        instruments=instruments,
        model_name=model_name,
        too_short=too_short,
    )
