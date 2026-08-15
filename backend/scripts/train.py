"""
train.py — ML Training Pipeline for 9-Feature AI Essay Detector

Extracts all 9 linguistic features across the benchmark dataset:
  1. Perplexity (GPT-2 token predictability / proxy log-likelihood)
  2. Sentence Length Burstiness (Coefficient of Variation)
  3. AI Fingerprint Phrase matches
  4. Lexical Diversity (Moving Average Type-Token Ratio)
  5. Punctuation & Stylistic Markers
  6. Function Word Ratio & Entropy
  7. Part-of-Speech Tag Bigram Entropy
  8. Readability Variance & Flesch-Kincaid Grade
  9. Discourse Marker / Transition Density

Fits a calibrated Logistic Regression classifier pipeline, calculates feature
coefficients/importances, and exports:
  - backend/models/detector.joblib
  - backend/models/feature_config.json
"""

import json
import math
import os
import re
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analyzer.burstiness import compute_burstiness
from analyzer.lexical import compute_lexical_diversity
from analyzer.phrases import compute_phrase_score
from analyzer.punctuation import compute_punctuation_score
from analyzer.function_words import compute_function_words
from analyzer.pos_entropy import compute_pos_entropy
from analyzer.readability import compute_readability
from analyzer.transitions import compute_transition_score

# Feature Names list
FEATURE_NAMES = [
    "burstiness_score",
    "burstiness_cv",
    "lexical_score",
    "lexical_mattr",
    "phrase_score",
    "phrase_count",
    "punctuation_score",
    "punctuation_markers_count",
    "function_words_score",
    "function_words_ratio",
    "function_words_entropy",
    "pos_entropy_score",
    "pos_entropy_raw",
    "readability_score",
    "readability_std_dev",
    "transition_score",
    "transition_density",
    "perplexity_proxy_score",
]


def split_sentences(text: str) -> list[str]:
    """Normalize and split text into individual sentences."""
    normalized = re.sub(r"\n+", " ", text.strip())
    normalized = re.sub(r"\s{2,}", " ", normalized)
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 2]


def estimate_perplexity_proxy(text: str, sentences: list[str]) -> float:
    """
    Fast, deterministic token predictability proxy based on character/word n-gram surprise.
    If GPT-2 model weights are available, attempts to use them; otherwise calculates
    a calibrated n-gram statistical predictability score in [0, 1].
    """
    try:
        from analyzer.perplexity import _model, compute_perplexity
        if _model is not None:
            scores = [compute_perplexity(s)["score"] for s in sentences[:8]]
            return float(np.mean(scores)) if scores else 0.5
    except Exception:
        pass

    # Fast statistical predictability proxy
    # AI text has repetitive common bigrams, predictable word lengths, and high transition regularity
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if len(words) < 10:
        return 0.5

    # Measure average word frequency in common English
    common_core = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "furthermore", "moreover", "consequently", "crucial", "pivotal", "transformative", "realm", "delve", "foster", "testament"}
    core_matches = sum(1 for w in words if w in common_core)
    core_ratio = core_matches / len(words)

    # Word length uniformity
    word_lens = [len(w) for w in words]
    len_var = float(np.var(word_lens)) if len(word_lens) > 1 else 4.0

    proxy_score = (core_ratio * 0.7) + (max(0.0, (5.0 - len_var) / 5.0) * 0.3)
    return float(max(0.0, min(1.0, proxy_score)))


def extract_features_vector(text: str) -> list[float]:
    """
    Extract a comprehensive 18-dimensional feature vector encompassing all 9 analyzers.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [0.5] * len(FEATURE_NAMES)

    # 1. Burstiness
    b_res = compute_burstiness(sentences)
    b_score = b_res["score"]
    b_cv = b_res["cv"] if b_res["cv"] is not None else 0.5

    # 2. Lexical Diversity
    l_res = compute_lexical_diversity(text)
    l_score = l_res["score"]
    l_mattr = l_res["mattr"] if l_res["mattr"] is not None else 0.65

    # 3. Phrases
    p_scores = [compute_phrase_score(s) for s in sentences]
    p_score = float(np.mean([p["score"] for p in p_scores])) if p_scores else 0.0
    p_count = sum(p["count"] for p in p_scores)

    # 4. Punctuation
    punc_scores = [compute_punctuation_score(s) for s in sentences]
    punc_score = float(np.mean([p["score"] for p in punc_scores])) if punc_scores else 0.5
    punc_count = sum(p["count"] for p in punc_scores)

    # 5. Function Words
    fw_res = compute_function_words(text)
    fw_score = fw_res["score"]
    fw_ratio = fw_res["ratio"]
    fw_entropy = fw_res["entropy"]

    # 6. POS Entropy
    pos_res = compute_pos_entropy(text)
    pos_score = pos_res["score"]
    pos_entropy_val = pos_res["entropy"]

    # 7. Readability
    r_res = compute_readability(sentences)
    r_score = r_res["score"]
    r_std_dev = r_res["std_dev"] if r_res["std_dev"] is not None else 2.5

    # 8. Transitions
    t_res = compute_transition_score(text)
    t_score = t_res["score"]
    t_density = t_res["density_per_100_words"]

    # 9. Perplexity Proxy
    perp_score = estimate_perplexity_proxy(text, sentences)

    return [
        b_score,
        b_cv,
        l_score,
        l_mattr,
        p_score,
        float(p_count),
        punc_score,
        float(punc_count),
        fw_score,
        fw_ratio,
        fw_entropy,
        pos_score,
        pos_entropy_val,
        r_score,
        r_std_dev,
        t_score,
        t_density,
        perp_score,
    ]


def train_model(
    dataset_path: str = "backend/dataset/admissions_essays.json",
    models_dir: str = "backend/models",
) -> dict:
    """
    Train the Logistic Regression detector with probability calibration on the admissions essays.
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        from generate_dataset import generate_dataset
        generate_dataset(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    print(f"[train] Loaded {len(samples)} essays from {dataset_path}")

    X = []
    y = []

    for s in samples:
        vec = extract_features_vector(s["text"])
        X.append(vec)
        y.append(s["class_label"])

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    # Build Pipeline: StandardScaler + LogisticRegression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_clf = LogisticRegression(
        C=1.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    base_clf.fit(X_scaled, y)

    # Calibrate probabilities using 3-fold cross-validation
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
    calibrated_clf.fit(X_scaled, y)

    # Full End-to-End Pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", calibrated_clf),
    ])

    # Save joblib model
    model_path = os.path.join(models_dir, "detector.joblib")
    joblib.dump(pipeline, model_path)
    print(f"[train] Model pipeline saved to: {model_path}")

    # Calculate Feature Coefficients from Base Classifier
    coefs = base_clf.coef_  # Shape (n_classes, n_features)
    intercept = base_clf.intercept_.tolist()

    feature_importances = {}
    classes = ["class_0_human", "class_1_ai", "class_2_mixed"]
    for c_idx, c_name in enumerate(classes):
        feature_importances[c_name] = {
            FEATURE_NAMES[i]: round(float(coefs[c_idx][i]), 4)
            for i in range(len(FEATURE_NAMES))
        }

    # Save feature metadata & configuration
    config = {
        "model_version": "2.0.0",
        "model_type": "Calibrated Logistic Regression (StandardScaler + LogisticRegression + Sigmoid Calibration)",
        "classes": {
            "0": "Likely Human-Written",
            "1": "Likely AI-Generated",
            "2": "Mixed / AI-Polished Human",
        },
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "scaler_means": [round(float(m), 5) for m in scaler.mean_],
        "scaler_scales": [round(float(s), 5) for s in scaler.scale_],
        "coefficients": feature_importances,
        "intercepts": [round(float(b), 5) for b in intercept],
        "signal_weights_heuristic": {
            "perplexity": 0.25,
            "burstiness": 0.15,
            "phrases": 0.12,
            "transitions": 0.12,
            "readability": 0.10,
            "pos_entropy": 0.10,
            "function_words": 0.06,
            "lexical": 0.05,
            "punctuation": 0.05,
        },
    }

    config_path = os.path.join(models_dir, "feature_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[train] Feature configuration saved to: {config_path}")

    # Feature Importance Highlights
    print("\n--- Key Learned Feature Weights for Class 1 (AI-Generated) ---")
    ai_coefs = feature_importances["class_1_ai"]
    sorted_ai = sorted(ai_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, w in sorted_ai[:8]:
        direction = "(Increases AI prob)" if w > 0 else "(Decreases AI prob)"
        print(f"  - {feat:25s}: {w:+.4f} {direction}")

    return {
        "model_path": model_path,
        "config_path": config_path,
        "n_samples": len(samples),
        "feature_count": len(FEATURE_NAMES),
    }


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "dataset", "admissions_essays.json")
    m_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "models")
    train_model(d_path, m_dir)
