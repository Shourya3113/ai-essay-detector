#!/usr/bin/env python3
"""Train a transparent sentence classifier, then calibrate essay scores.

Order: logistic regression baseline → LightGBM if it installs and beats
F1 without looking wildly overfit → Platt/isotonic calibration.

Never calls a chat model.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.extractor import FEATURE_NAMES, extract_essay, vectorize  # noqa: E402
from features.nlp_resources import skip_neural  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SEED = 14


def load_essays(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_for_span(start: int, end: int, anns: list[dict]) -> int:
    """Majority overlap: ai=1, human=0."""
    best_ai = 0.0
    best_h = 0.0
    for a in anns:
        lo = max(start, a["start"])
        hi = min(end, a["end"])
        overlap = max(0, hi - lo)
        if a.get("label") == "ai":
            best_ai += overlap
        else:
            best_h += overlap
    if best_ai + best_h == 0:
        return 0
    return 1 if best_ai >= best_h else 0


def build_matrix(essays: list[dict], cache_path: Path | None = None):
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            np.array(data["X"], dtype=float),
            np.array(data["y"], dtype=int),
            data["essay_ids"],
            np.array(data["essay_y"], dtype=int),
        )

    X: list[list[float]] = []
    y: list[int] = []
    essay_ids: list[str] = []
    essay_y: list[int] = []
    for i, essay in enumerate(essays):
        rec = extract_essay(essay["text"], use_neural=None)
        anns = essay.get("sentence_annotations") or []
        essay_has_ai = 1 if essay.get("essay_label") in {"machine", "mixed"} else 0
        print(f"  features {i + 1}/{len(essays)}  id={essay.get('id')}  sents={len(rec.sentences)}")
        for sent in rec.sentences:
            X.append(vectorize(sent))
            y.append(label_for_span(sent.span.start, sent.span.end, anns))
            essay_ids.append(essay["id"])
            essay_y.append(essay_has_ai)
    arr_x, arr_y = np.array(X, dtype=float), np.array(y, dtype=int)
    if cache_path:
        cache_path.write_text(
            json.dumps(
                {
                    "X": arr_x.tolist(),
                    "y": arr_y.tolist(),
                    "essay_ids": essay_ids,
                    "essay_y": essay_y,
                    "feature_names": list(FEATURE_NAMES),
                }
            ),
            encoding="utf-8",
        )
    return arr_x, arr_y, essay_ids, np.array(essay_y, dtype=int)


def essay_split(essay_ids: list[str], essay_y: np.ndarray, test_size: float = 0.28):
    """Split by essay, not sentence, so neighbors don't leak."""
    unique = []
    seen = set()
    labels = []
    for eid, ey in zip(essay_ids, essay_y):
        if eid in seen:
            continue
        seen.add(eid)
        unique.append(eid)
        labels.append(int(ey))
    try:
        tr, te = train_test_split(
            unique, test_size=test_size, random_state=SEED, stratify=labels
        )
    except ValueError:
        tr, te = train_test_split(unique, test_size=test_size, random_state=SEED)
    tr_s, te_s = set(tr), set(te)
    return tr_s, te_s


def mask_for(ids: list[str], keep: set[str]) -> np.ndarray:
    return np.array([i in keep for i in ids], dtype=bool)


def try_hist_gb(Xtr, ytr, Xte, yte):
    """sklearn stand-in when LightGBM's libomp is missing."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        print("HistGradientBoosting unavailable:", exc)
        return None, None
    clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=80, learning_rate=0.08, random_state=SEED
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, zero_division=0)
    print("HistGradientBoosting sentence F1:", round(f1, 3))
    return clf, f1


def try_lightgbm(Xtr, ytr, Xte, yte):
    try:
        import lightgbm as lgb
    except Exception as exc:
        print("LightGBM unavailable:", exc)
        return None, None
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_samples=3,
        random_state=SEED,
        verbose=-1,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, zero_division=0)
    print("LightGBM sentence F1:", round(f1, 3))
    return clf, f1


def fit_isotonic(p_train: np.ndarray, y_train: np.ndarray) -> IsotonicRegression | None:
    if len(np.unique(y_train)) < 2 or len(y_train) < 8:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    return iso


def main() -> None:
    essays_path = PROCESSED / "essays.jsonl"
    if not essays_path.exists():
        raise SystemExit("Run python scripts/generate_dataset.py first.")
    essays = load_essays(essays_path)
    print(f"Loaded {len(essays)} essays")
    MODELS.mkdir(parents=True, exist_ok=True)
    X, y, eids, ey = build_matrix(essays, PROCESSED / "feature_cache.json")
    print("Sentence label counts:", dict(Counter(y.tolist())))

    tr_essays, te_essays = essay_split(eids, ey)
    tr, te = mask_for(eids, tr_essays), mask_for(eids, te_essays)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"Train sentences {len(ytr)} / test sentences {len(yte)}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    logreg = LogisticRegression(max_iter=400, class_weight="balanced", C=0.8)
    # Nested calibration if we have enough rows.
    if len(ytr) >= 40:
        sentence_clf = CalibratedClassifierCV(logreg, method="sigmoid", cv=3)
    else:
        sentence_clf = logreg
    sentence_clf.fit(Xtr_s, ytr)
    pred_lr = sentence_clf.predict(Xte_s)
    f1_lr = f1_score(yte, pred_lr, zero_division=0)
    print("LogReg sentence F1:", round(f1_lr, 3))
    print(classification_report(yte, pred_lr, zero_division=0, target_names=["human", "ai"]))
    print("Confusion matrix:\n", confusion_matrix(yte, pred_lr))

    lgbm, f1_lgb = try_lightgbm(Xtr_s, ytr, Xte_s, yte)
    hgb, f1_hgb = try_hist_gb(Xtr_s, ytr, Xte_s, yte)
    kind = "logreg"
    chosen = sentence_clf
    # Keep logistic regression unless a tree model wins *and* we can still
    # explain flags. On this repo we prefer LR for signed contributions.
    print("Selected logistic regression for explainability (tree models logged above).")

    def predict_p(clf, Xn) -> np.ndarray:
        if kind == "lightgbm":
            raw = clf.predict(Xn, raw_score=True)
            return 1.0 / (1.0 + np.exp(-np.ravel(raw)))
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xn)[:, 1]
        return clf.decision_function(Xn)

    p_tr = predict_p(chosen, Xtr_s)
    p_te = predict_p(chosen, Xte_s)

    # Essay-level: mean sentence proba, calibrated against essay label.
    def essay_means(ids, mask, probs, essay_labels):
        buckets: dict[str, list[float]] = {}
        labels: dict[str, int] = {}
        for eid, keep, p, lab in zip(ids, mask, probs, essay_labels):
            if not keep:
                continue
            buckets.setdefault(eid, []).append(float(p))
            labels[eid] = int(lab)
        eids_u = list(buckets)
        means = np.array([float(np.mean(buckets[e])) for e in eids_u])
        ys = np.array([labels[e] for e in eids_u])
        return eids_u, means, ys

    # Need train sentence probs aligned to train mask.
    p_all = np.zeros(len(y))
    p_all[tr] = p_tr
    p_all[te] = p_te
    _, mean_tr, y_ess_tr = essay_means(eids, tr, p_all, ey)
    _, mean_te, y_ess_te = essay_means(eids, te, p_all, ey)
    essay_cal = fit_isotonic(mean_tr, y_ess_tr)

    importances = {}
    if kind == "logreg":
        base = chosen
        if hasattr(base, "calibrated_classifiers_"):
            coefs = np.mean(
                [
                    np.ravel(
                        getattr(c, "estimator", getattr(c, "base_estimator", c)).coef_
                    )
                    for c in base.calibrated_classifiers_
                ],
                axis=0,
            )
        elif hasattr(base, "coef_"):
            coefs = np.ravel(base.coef_)
        else:
            coefs = np.zeros(len(FEATURE_NAMES))
        importances = {n: float(c) for n, c in zip(FEATURE_NAMES, coefs)}
    elif kind == "lightgbm":
        importances = {
            n: float(v) for n, v in zip(FEATURE_NAMES, chosen.feature_importances_)
        }

    bundle = {
        "kind": kind,
        "model_name": f"{kind}-calibrated",
        "skip_neural": skip_neural(),
        "sentence_clf": chosen,
        "scaler": scaler,
        "essay_calibrator": essay_cal,
        "feature_names": list(FEATURE_NAMES),
        "importances": importances,
        "test_essay_ids": sorted(te_essays),
        "metrics": {
            "sentence_f1_logreg": float(f1_lr),
            "sentence_f1_lightgbm": float(f1_lgb) if f1_lgb is not None else None,
            "sentence_f1_histgb": float(f1_hgb) if f1_hgb is not None else None,
            "chosen": kind,
            "n_train_sentences": int(len(ytr)),
            "n_test_sentences": int(len(yte)),
            "n_train_essays": int(len(tr_essays)),
            "n_test_essays": int(len(te_essays)),
        },
    }
    out = MODELS / "detector.joblib"
    joblib.dump(bundle, out)
    (MODELS / "feature_importances.json").write_text(
        json.dumps(importances, indent=2), encoding="utf-8"
    )
    split_doc = {
        "test_essay_ids": sorted(te_essays),
        "train_essay_ids": sorted(tr_essays),
    }
    (PROCESSED / "split.json").write_text(json.dumps(split_doc, indent=2), encoding="utf-8")
    print("Saved", out)
    print("Top |importances|:")
    for k, v in sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
        print(f"  {k:28s} {v:+.4f}")


if __name__ == "__main__":
    main()
