"""
scorer.py — 9-Feature Linguistic Scoring & ML Calibration Orchestrator

Combines 9 independent linguistic signals into an explainable, calibrated AI probability score:

Features:
  1. Perplexity (GPT-2 log-likelihood / token predictability)
  2. Sentence Length Burstiness (Coefficient of Variation)
  3. AI Fingerprint Phrases (Overrepresented n-grams)
  4. Lexical Diversity (Moving Average Type-Token Ratio)
  5. Punctuation & Stylistic Irregularities (Em-dashes, parentheticals, etc.)
  6. Function Word Ratio & Entropy (Grammatical glue distribution)
  7. POS Bigram Transition Entropy (Syntactic template regularity)
  8. Readability Variance (Sentence-by-sentence Flesch-Kincaid variance)
  9. Academic Discourse Transitions (Frequency & clustering of connectives)

ML Integration:
  Loads calibrated `backend/models/detector.joblib` when available.
  Falls back smoothly to `backend/models/feature_config.json` (calibrated logistic
  regression inference) or the 9-feature weighted heuristic blend.
"""

import json
import math
import os
import re
from fastapi import HTTPException

# Feature Analyzers
from analyzer.perplexity import compute_perplexity
from analyzer.burstiness import compute_burstiness
from analyzer.lexical import compute_lexical_diversity
from analyzer.phrases import compute_phrase_score
from analyzer.punctuation import compute_punctuation_score
from analyzer.function_words import compute_function_words
from analyzer.pos_entropy import compute_pos_entropy
from analyzer.readability import compute_readability, compute_sentence_readability
from analyzer.transitions import compute_transition_score

# Paths to ML artifacts
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "detector.joblib"))
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "feature_config.json"))

_ml_pipeline = None
_model_loaded_attempted = False
_feature_config = None


def load_feature_config() -> dict | None:
    """Load feature configuration containing learned scaler and regression weights."""
    global _feature_config
    if _feature_config is not None:
        return _feature_config

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _feature_config = json.load(f)
        except Exception as e:
            print(f"[scorer] Error loading feature config: {e}")
            _feature_config = None
    return _feature_config


def load_ml_pipeline():
    """Attempt to load the trained ML pipeline from disk."""
    global _ml_pipeline, _model_loaded_attempted
    if _model_loaded_attempted:
        return _ml_pipeline

    _model_loaded_attempted = True
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            _ml_pipeline = joblib.load(MODEL_PATH)
            print(f"[scorer] Loaded calibrated ML detector model from {MODEL_PATH}")
        except Exception as e:
            print(f"[scorer] Could not load ML joblib model ({e}). Using feature_config weights.")
            _ml_pipeline = None
    else:
        _ml_pipeline = None
    return _ml_pipeline


def predict_with_feature_config(feature_vector: list[float], config: dict) -> dict:
    """
    Compute calibrated multi-class probabilities using exported model weights and scaler parameters.
    """
    means = config["scaler_means"]
    scales = config["scaler_scales"]
    coefs_dict = config["coefficients"]
    intercepts = config["intercepts"]
    feat_names = config["feature_names"]

    # 1. Standardize features
    x_scaled = []
    for i in range(len(feature_vector)):
        val = feature_vector[i]
        m = means[i] if i < len(means) else 0.0
        s = scales[i] if (i < len(scales) and scales[i] > 1e-6) else 1.0
        x_scaled.append((val - m) / s)

    # 2. Compute logits for each class
    logits = []
    class_keys = ["class_0_human", "class_1_ai", "class_2_mixed"]
    for idx, ckey in enumerate(class_keys):
        w_map = coefs_dict.get(ckey, {})
        dot_product = sum(x_scaled[j] * w_map.get(feat_names[j], 0.0) for j in range(len(feat_names)))
        b = intercepts[idx] if idx < len(intercepts) else 0.0
        logits.append(dot_product + b)

    # 3. Softmax
    max_logit = max(logits)
    exp_logits = [math.exp(z - max_logit) for z in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    return {
        "human": round(float(probs[0]), 4),
        "ai": round(float(probs[1]), 4),
        "mixed": round(float(probs[2]), 4),
    }


def split_sentences(text: str) -> list[str]:
    """Normalize whitespace and split text into distinct sentences."""
    normalized = re.sub(r"\n+", " ", text.strip())
    normalized = re.sub(r"\s{2,}", " ", normalized)
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 2]


def generate_sentence_explanations(
    sentence: str,
    perp_res: dict,
    phrase_res: dict,
    punc_res: dict,
    readability_res: dict,
    trans_res: dict,
    func_res: dict,
    mean_sent_len: float,
) -> list[str]:
    """
    Generate transparent, human-readable linguistic explanations for why
    this specific sentence leans AI or Human.
    """
    explanations = []

    # 1. Perplexity reasons
    if perp_res.get("raw") is not None:
        if perp_res["raw"] < 50:
            explanations.append(f"Low Perplexity ({perp_res['raw']:.0f}) — highly predictable to language models")
        elif perp_res["raw"] > 180:
            explanations.append(f"High Perplexity ({perp_res['raw']:.0f}) — surprising / human-like word choices")

    # 2. Phrase fingerprints
    if phrase_res.get("matched"):
        matched_names = [m["phrase"] for m in phrase_res["matched"][:2]]
        explanations.append(f"AI Fingerprint Phrase: \"{', '.join(matched_names)}\"")

    # 3. Transitions & Discourse markers
    if trans_res.get("matched"):
        t_names = [m["phrase"] for m in trans_res["matched"][:2]]
        explanations.append(f"Cliché Transition: \"{', '.join(t_names)}\"")

    # 4. Stylistic punctuation
    if punc_res.get("markers"):
        markers = punc_res["markers"][:2]
        explanations.append(f"Human Stylistic Marker: {', '.join(markers)}")

    # 5. Sentence length uniformity vs variance
    words = sentence.split()
    sent_len = len(words)
    if mean_sent_len > 0 and abs(sent_len - mean_sent_len) <= 2 and sent_len >= 14:
        explanations.append("Uniform Sentence Length (standard AI pacing)")
    elif sent_len <= 5:
        explanations.append("Punchy short sentence (human narrative pacing)")

    # 6. Readability
    if readability_res.get("fk_grade") is not None:
        if readability_res["fk_grade"] >= 15.0:
            explanations.append(f"Elevated complexity (Grade {readability_res['fk_grade']:.1f})")

    # Fallback explanation if empty
    if not explanations:
        if perp_res.get("score", 0.5) >= 0.6:
            explanations.append("Standard grammatical structure and predictable phrasing")
        else:
            explanations.append("Natural expressive phrasing")

    return explanations


def analyze_essay(text: str) -> dict:
    """
    Full 9-feature document and sentence-level analysis with calibrated ML inference.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Essay text cannot be empty.")

    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found in text.")

    # -------------------------------------------------------------------------
    # 1. DOCUMENT-LEVEL SIGNALS
    # -------------------------------------------------------------------------
    burstiness_res = compute_burstiness(sentences)
    lexical_res = compute_lexical_diversity(text)
    readability_res = compute_readability(sentences)
    pos_res = compute_pos_entropy(text)
    function_words_res = compute_function_words(text)
    transitions_res = compute_transition_score(text)

    mean_len = burstiness_res.get("mean_len") or 0.0

    # -------------------------------------------------------------------------
    # 2. SENTENCE-LEVEL SIGNALS & EXPLANATIONS
    # -------------------------------------------------------------------------
    sentence_results = []
    total_heuristic_score = 0.0

    for idx, sentence in enumerate(sentences):
        perp_res = compute_perplexity(sentence)
        phrase_res = compute_phrase_score(sentence)
        punc_res = compute_punctuation_score(sentence)
        sent_readability_res = compute_sentence_readability(sentence)
        sent_trans_res = compute_transition_score(sentence)
        sent_func_res = compute_function_words(sentence)

        # Multi-factor sentence score formula combining all 9 signals
        sentence_score = (
            0.25 * perp_res["score"]
            + 0.15 * burstiness_res["score"]
            + 0.12 * phrase_res["score"]
            + 0.12 * sent_trans_res["score"]
            + 0.10 * readability_res["score"]
            + 0.10 * pos_res["score"]
            + 0.06 * sent_func_res["score"]
            + 0.05 * lexical_res["score"]
            + 0.05 * punc_res["score"]
        )

        sentence_score = float(max(0.0, min(1.0, sentence_score)))
        total_heuristic_score += sentence_score

        explanations = generate_sentence_explanations(
            sentence=sentence,
            perp_res=perp_res,
            phrase_res=phrase_res,
            punc_res=punc_res,
            readability_res=sent_readability_res,
            trans_res=sent_trans_res,
            func_res=sent_func_res,
            mean_sent_len=mean_len,
        )

        sentence_results.append({
            "id": idx,
            "text": sentence,
            "score": round(sentence_score, 4),
            "explanations": explanations,
            "signals": {
                "perplexity": perp_res,
                "phrases": phrase_res,
                "punctuation": punc_res,
                "readability": sent_readability_res,
                "transitions": sent_trans_res,
                "function_words": sent_func_res,
            },
        })

    heuristic_overall_score = total_heuristic_score / len(sentences)

    # -------------------------------------------------------------------------
    # 3. ML MODEL PREDICTION (Calibrated Logistic Regression)
    # -------------------------------------------------------------------------
    pipeline = load_ml_pipeline()
    config = load_feature_config()
    ml_calibrated = False
    class_probs = {"human": 0.0, "ai": 0.0, "mixed": 0.0}

    # Extract 18-feature vector
    b_score = burstiness_res["score"]
    b_cv = burstiness_res["cv"] if burstiness_res["cv"] is not None else 0.5
    l_score = lexical_res["score"]
    l_mattr = lexical_res["mattr"] if lexical_res["mattr"] is not None else 0.65
    p_score = (
        sum(s["signals"]["phrases"]["score"] for s in sentence_results) / len(sentence_results)
        if sentence_results else 0.0
    )
    p_count = sum(s["signals"]["phrases"]["count"] for s in sentence_results)
    punc_score = (
        sum(s["signals"]["punctuation"]["score"] for s in sentence_results) / len(sentence_results)
        if sentence_results else 0.5
    )
    punc_count = sum(s["signals"]["punctuation"]["count"] for s in sentence_results)
    fw_score = function_words_res["score"]
    fw_ratio = function_words_res["ratio"]
    fw_entropy = function_words_res["entropy"]
    pos_score = pos_res["score"]
    pos_entropy_val = pos_res["entropy"]
    r_score = readability_res["score"]
    r_std_dev = readability_res["std_dev"] if readability_res["std_dev"] is not None else 2.5
    t_score = transitions_res["score"]
    t_density = transitions_res["density_per_100_words"]
    perp_mean = (
        sum(s["signals"]["perplexity"]["score"] for s in sentence_results) / len(sentence_results)
        if sentence_results else 0.5
    )

    feature_vector = [
        b_score, b_cv, l_score, l_mattr, p_score, float(p_count),
        punc_score, float(punc_count), fw_score, fw_ratio, fw_entropy,
        pos_score, pos_entropy_val, r_score, r_std_dev, t_score, t_density,
        perp_mean,
    ]

    if pipeline is not None:
        try:
            probs = pipeline.predict_proba([feature_vector])[0]
            class_probs = {
                "human": round(float(probs[0]), 4),
                "ai": round(float(probs[1]), 4),
                "mixed": round(float(probs[2]), 4),
            }
            overall_score = float(probs[1] + (0.5 * probs[2]))
            overall_score = max(0.0, min(1.0, overall_score))
            ml_calibrated = True
        except Exception as err:
            print(f"[scorer] Joblib prediction fallback: {err}")
            pipeline = None

    if not ml_calibrated and config is not None:
        try:
            class_probs = predict_with_feature_config(feature_vector, config)
            overall_score = float(class_probs["ai"] + (0.5 * class_probs["mixed"]))
            overall_score = max(0.0, min(1.0, overall_score))
            ml_calibrated = True
        except Exception as err:
            print(f"[scorer] Feature config prediction fallback: {err}")

    if not ml_calibrated:
        overall_score = heuristic_overall_score
        if overall_score >= 0.70:
            class_probs = {"human": round(max(0.0, 1.0 - overall_score), 4), "ai": round(overall_score, 4), "mixed": round(0.15, 4)}
        elif overall_score >= 0.45:
            class_probs = {"human": 0.25, "ai": 0.25, "mixed": 0.50}
        else:
            class_probs = {"human": round(1.0 - overall_score, 4), "ai": round(overall_score, 4), "mixed": round(0.10, 4)}

    # -------------------------------------------------------------------------
    # 4. CLASSIFICATION VERDICT
    # -------------------------------------------------------------------------
    if overall_score >= 0.68:
        verdict = "Likely AI-Generated"
    elif overall_score >= 0.42:
        verdict = "Mixed / AI-Polished Human"
    else:
        verdict = "Likely Human-Written"

    return {
        "overall_score": round(overall_score, 4),
        "verdict": verdict,
        "ml_calibrated": ml_calibrated,
        "class_probabilities": class_probs,
        "document_signals": {
            "burstiness": burstiness_res,
            "lexical_diversity": lexical_res,
            "readability": readability_res,
            "pos_entropy": pos_res,
            "function_words": function_words_res,
            "transitions": transitions_res,
        },
        "sentences": sentence_results,
        "diagnostics": {
            "features_extracted": 9,
            "total_sentences": len(sentences),
            "ml_model_active": ml_calibrated,
        },
    }
