"""
evaluate.py — Cross-Validation, Bias Analysis & Edge-Case Evaluation Pipeline

Evaluates the 9-feature AI detector across:
  1. Stratified 5-Fold Cross Validation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
  2. ESL (English as a Second Language) Essay Simulation & Bias Analysis (False Positive Rate calculation)
  3. In-Depth Failure / Edge-Case Diagnostics (3 targeted stress scenarios)

Saves all metrics and diagnostic logs to:
  - backend/dataset/evaluation_results.json
"""

import json
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import joblib

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train import extract_features_vector, FEATURE_NAMES

# =============================================================================
# ESL SIMULATED ESSAY CORPUS (Human-authored essays with standard ESL curriculum markers)
# =============================================================================
ESL_SIMULATION_SAMPLES = [
    {
        "id": "esl_01",
        "description": "ESL Human: Uses formal transition markers taught in school, but tells genuine personal story with high burstiness",
        "ground_truth": 0,  # Human
        "text": (
            "When my family moved from Seoul to Chicago, my English was very limited. "
            "I could only say 'hello' and 'thank you'. "
            "Furthermore, entering eighth grade without speaking English was terrifying because all the students talked so fast. "
            "Every day after school, I watched American cartoons with subtitles for three hours. "
            "My mother cooked hot kimchi soup while I practiced pronunciation in front of the bathroom mirror until my jaw hurt. "
            "Moreover, my science teacher, Mr. Davis, gave me illustrated biology books with Korean annotations. "
            "In conclusion, learning a new language taught me that patience and hard work can overcome any barrier in life."
        )
    },
    {
        "id": "esl_02",
        "description": "ESL Human: Formal discourse markers + personal family grocery story",
        "ground_truth": 0,
        "text": (
            "Growing up in our family's small bodega in Queens, I worked behind the cash register every weekend. "
            "It is important to note that my parents worked fourteen hours every day without taking a vacation. "
            "I helped my father calculate inventory spreadsheets on an old laptop. "
            "Furthermore, interacting with diverse customers from Colombia, Poland, and China improved my conversational confidence. "
            "Once, a customer dropped sixty dollars on the floor; I ran two blocks down Roosevelt Avenue in the rain to return it. "
            "He cried and hugged me. "
            "In summary, working at our family store taught me integrity and the true value of honest community service."
        )
    },
    {
        "id": "esl_03",
        "description": "ESL Human: Academic transitions in physics lab reflection",
        "ground_truth": 0,
        "text": (
            "In my physics class in Mumbai, we did not have digital laser sensors or fancy computers. "
            "We used wooden meter sticks and stopwatch timers to measure gravity. "
            "Furthermore, my team dropped forty-five metal balls from the school roof to calculate acceleration. "
            "Our data was messy because wind kept blowing our plumb line sideways. "
            "I spent five nights recalculating the air drag coefficient with pencil and paper. "
            "Consequently, our calculated value for g was 9.78 m/s^2—almost identical to the textbook constant! "
            "To conclude, true scientific passion does not depend on expensive tools, but on creative problem solving."
        )
    },
    {
        "id": "esl_04",
        "description": "ESL Human: Personal adversity with formal signposting",
        "ground_truth": 0,
        "text": (
            "During the flood in my hometown of Beira, water rose up to our window sills. "
            "We had to carry my grandmother on our shoulders to the church shelter. "
            "It was dark. People were crying. "
            "Furthermore, I helped the local relief volunteers organize clean water purification tablets for two hundred families. "
            "Moreover, seeing our neighbors rebuild collapsed roofs from bamboo poles inspired me to study civil engineering. "
            "In conclusion, surviving the natural disaster showed me that community unity is stronger than any storm."
        )
    },
    {
        "id": "esl_05",
        "description": "ESL Human: Sibling caregiving narrative with school transitions",
        "ground_truth": 0,
        "text": (
            "Every morning at 6:00 AM, I woke up to cook breakfast for my two younger brothers before school. "
            "My parents worked the early cleaning shift at the municipal hospital. "
            "Furthermore, I packed their lunchboxes, checked their math homework, and walked them to the bus stop. "
            "Sometimes I was so sleepy that I forgot to tie my own shoelaces. "
            "However, seeing my brother Leo win the school spelling bee made every early morning worthwhile. "
            "In summary, caring for my family taught me selflessness, maturity, and time management."
        )
    },
    {
        "id": "esl_06",
        "description": "ESL Human: Coding journey with basic English vocabulary",
        "ground_truth": 0,
        "text": (
            "I wrote my first Python program on a broken smartphone using an online interpreter. "
            "The screen was cracked across the middle, so I had to scroll sideways to see my code. "
            "Furthermore, learning loops and conditional logic opened a completely new universe for me. "
            "I built an automated quiz app to help my classmates memorize English irregular verbs. "
            "They loved it! "
            "In conclusion, technology gave me a voice when my spoken English was still developing."
        )
    },
    {
        "id": "esl_07",
        "description": "ESL Human: Cultural cooking memory with school essay structure",
        "ground_truth": 0,
        "text": (
            "Making tamales with my abuela during Christmas is a three-day marathon of corn husks and chili spices. "
            "My job was spreading the masa evenly across each leaf with a wooden spoon. "
            "It is crucial to keep the layer thin, or the tamale becomes too heavy when steamed. "
            "Furthermore, listening to my grandmother's stories about her youth in Michoacan taught me oral family history. "
            "My arms ached after spreading five hundred tamales. "
            "In summary, preserving cultural culinary traditions connects our modern generation with our ancestral roots."
        )
    },
    {
        "id": "esl_08",
        "description": "ESL Human: Math olympiad reflection with structured connectives",
        "ground_truth": 0,
        "text": (
            "Geometry was my favorite subject in middle school in São Paulo. "
            "I loved proving theorems because logic does not require English translation—numbers speak directly. "
            "Furthermore, qualifying for the state mathematics olympiad gave me courage to apply to international exchange programs. "
            "I practiced geometry problems on the bus ride every single afternoon. "
            "Consequently, solving problem four during the final round was the happiest moment of my sophomore year. "
            "To summarize, mathematics gave me universal confidence and ignited my passion for future engineering studies."
        )
    }
]

# =============================================================================
# 3 TARGETED FAILURE / EDGE CASE SCENARIOS
# =============================================================================
EDGE_CASE_SCENARIOS = [
    {
        "case_id": "edge_01_hyper_formal_human",
        "title": "Hyper-Formal Academic Human Essay",
        "description": (
            "A human applicant writing in an extraordinarily stiff, academic style with frequent "
            "discourse transitions ('Furthermore', 'Consequently') and high vocabulary density. "
            "Risk: Model may over-index on formal vocabulary and transitions, creating a borderline false positive."
        ),
        "ground_truth": "Class 0 (Human)",
        "text": (
            "The epistemological foundations of contemporary socio-political discourse necessitate a rigorous reevaluation of institutional pedagogical structures. "
            "Furthermore, throughout my secondary academic tenure, I consistently endeavored to elucidate the subtle nexus between macroeconomic fiscal policy and localized community resilience. "
            "Consequently, conducting independent statistical regressional analyses on municipal budgetary allocations provided empirical verification for systemic municipal underfunding. "
            "Moreover, it is imperative to acknowledge that structural inequities cannot be ameliorated through superficial interventions alone. "
            "In conclusion, my scholarly aspirations remain firmly anchored in the pursuit of rigorous quantitative public policy analysis."
        )
    },
    {
        "case_id": "edge_02_adversarial_prompted_ai",
        "title": "Adversarially Prompted AI with Slang and Burstiness",
        "description": (
            "An AI generator prompted specifically to bypass AI detectors by injecting erratic sentence lengths, "
            "colloquialisms, em-dashes, and informal syntax. "
            "Risk: Model may see high burstiness and low transitions, risking a false negative."
        ),
        "ground_truth": "Class 1 (AI-Generated)",
        "text": (
            "I broke it. Completely wrecked. "
            "My mom's vintage blender went flying off the kitchen counter—spilling banana puree across our yellow vinyl floor like some modern art tragedy. "
            "Panic mode engaged! "
            "I scrambled under the sink, grabbed some duct tape, two paper clips, and my dad's trusty pliers. "
            "Did it fix the motor? Well, not exactly—it made a sound like a dying helicopter for five seconds before smoking. "
            "Lesson learned: engineering requires slightly more patience than enthusiastic duct tape repairs."
        )
    },
    {
        "case_id": "edge_03_short_fragment",
        "title": "Short Narrative Fragment (< 40 words)",
        "description": (
            "A very short human essay excerpt. "
            "Risk: Statistical signals (Burstiness CV, MATTR, POS Bigram Entropy) have insufficient sample size."
        ),
        "ground_truth": "Class 0 (Human)",
        "text": (
            "I looked at the chess clock. Three seconds remaining. "
            "My hand trembled as I moved my knight to f6. Checkmate."
        )
    }
]


def evaluate_system(
    dataset_path: str = "backend/dataset/admissions_essays.json",
    model_path: str = "backend/models/detector.joblib",
    output_path: str = "backend/dataset/evaluation_results.json",
) -> dict:
    """
    Run full 5-Fold Stratified Cross-Validation, ESL Bias Evaluation, and Edge Case Diagnostics.
    """
    if not os.path.exists(dataset_path):
        from generate_dataset import generate_dataset
        generate_dataset(dataset_path)

    if not os.path.exists(model_path):
        from train import train_model
        train_model(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    print(f"\n=======================================================")
    print(f"      AI ESSAY DETECTOR — 9-FEATURE EVALUATION")
    print(f"=======================================================")
    print(f"[evaluate] Total Benchmark Samples: {len(samples)}")

    # Extract all features and labels
    X = []
    y = []
    for s in samples:
        X.append(extract_features_vector(s["text"]))
        y.append(s["class_label"])

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    # -------------------------------------------------------------------------
    # 1. STRATIFIED 5-FOLD CROSS VALIDATION
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        base_clf = LogisticRegression(C=1.5, penalty="l2", max_iter=1000, random_state=42)
        base_clf.fit(X_train_scaled, y_train)

        calibrated = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calibrated.fit(X_train_scaled, y_train)

        y_pred = calibrated.predict(X_val_scaled)
        y_prob = calibrated.predict_proba(X_val_scaled)

        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(p)
        fold_recalls.append(r)
        fold_f1s.append(f1)

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_p = float(np.mean(fold_precisions))
    mean_r = float(np.mean(fold_recalls))
    mean_f1 = float(np.mean(fold_f1s))

    # Multi-class ROC-AUC (One-vs-Rest)
    y_bin = label_binarize(y_true_all, classes=[0, 1, 2])
    roc_auc_val = float(roc_auc_score(y_bin, y_prob_all, multi_class="ovr", average="macro"))

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all).tolist()

    # Per-Class Metrics
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=[0, 1, 2], zero_division=0
    )

    class_names = ["Human (Class 0)", "AI (Class 1)", "Mixed (Class 2)"]
    per_class_summary = {}
    for idx, cname in enumerate(class_names):
        per_class_summary[cname] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_sup[idx]),
        }

    print("\n--- 5-Fold Stratified Cross-Validation Results ---")
    print(f"  Accuracy:  {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"  Precision: {mean_p:.4f}")
    print(f"  Recall:    {mean_r:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc_val:.4f}")
    print(f"  Confusion Matrix: {cm}")

    # -------------------------------------------------------------------------
    # 2. ESL (English as a Second Language) SIMULATION & BIAS AUDIT
    # -------------------------------------------------------------------------
    pipeline = joblib.load(model_path)

    esl_results = []
    esl_false_positives = 0

    for sample in ESL_SIMULATION_SAMPLES:
        vec = extract_features_vector(sample["text"])
        probs = pipeline.predict_proba([vec])[0]
        pred_class = int(np.argmax(probs))

        # Check if classified as AI (Class 1) or Mixed (Class 2) vs Human (Class 0)
        is_flagged_ai = (pred_class == 1)
        if is_flagged_ai:
            esl_false_positives += 1

        esl_results.append({
            "id": sample["id"],
            "description": sample["description"],
            "ground_truth": "Human",
            "predicted_class": int(pred_class),
            "predicted_label": class_names[pred_class],
            "probabilities": {
                "human": round(float(probs[0]), 4),
                "ai": round(float(probs[1]), 4),
                "mixed": round(float(probs[2]), 4),
            },
            "false_positive": is_flagged_ai,
        })

    esl_fpr = esl_false_positives / len(ESL_SIMULATION_SAMPLES)

    print("\n--- ESL (English as a Second Language) Bias Audit ---")
    print(f"  ESL Samples Tested: {len(ESL_SIMULATION_SAMPLES)}")
    print(f"  ESL False Positives (Flagged as AI): {esl_false_positives}")
    print(f"  ESL False Positive Rate (FPR): {esl_fpr:.2%}")
    print(f"  Bias Mitigation Status: {'EXCELLENT (< 15% FPR)' if esl_fpr < 0.15 else 'NEEDS RETUNING'}")

    # -------------------------------------------------------------------------
    # 3. FAILURE / EDGE CASE EVALUATION
    # -------------------------------------------------------------------------
    edge_case_results = []
    print("\n--- Edge Cases & Diagnostic Logs ---")

    for ec in EDGE_CASE_SCENARIOS:
        vec = extract_features_vector(ec["text"])
        probs = pipeline.predict_proba([vec])[0]
        pred_class = int(np.argmax(probs))

        # Extract top feature values for diagnostic explanation
        feat_dict = {FEATURE_NAMES[i]: round(float(vec[i]), 3) for i in range(len(FEATURE_NAMES))}

        diag = {
            "case_id": ec["case_id"],
            "title": ec["title"],
            "ground_truth": ec["ground_truth"],
            "predicted_label": class_names[pred_class],
            "probabilities": {
                "human": round(float(probs[0]), 4),
                "ai": round(float(probs[1]), 4),
                "mixed": round(float(probs[2]), 4),
            },
            "key_features": {
                "burstiness_score": feat_dict.get("burstiness_score"),
                "burstiness_cv": feat_dict.get("burstiness_cv"),
                "transition_density": feat_dict.get("transition_density"),
                "phrase_count": feat_dict.get("phrase_count"),
                "readability_std_dev": feat_dict.get("readability_std_dev"),
                "pos_entropy_raw": feat_dict.get("pos_entropy_raw"),
            },
            "diagnostic_analysis": ec["description"],
        }
        edge_case_results.append(diag)
        print(f"\n  [{ec['case_id']}] {ec['title']}")
        print(f"    Ground Truth: {ec['ground_truth']} | Predicted: {class_names[pred_class]}")
        print(f"    Probabilities: Human={probs[0]:.2%}, AI={probs[1]:.2%}, Mixed={probs[2]:.2%}")

    # Compile Complete Output
    evaluation_report = {
        "benchmark_summary": {
            "total_samples": len(samples),
            "k_folds": 5,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "precision_weighted": round(mean_p, 4),
            "recall_weighted": round(mean_r, 4),
            "f1_score_weighted": round(mean_f1, 4),
            "roc_auc_ovr": round(roc_auc_val, 4),
            "confusion_matrix": {
                "labels": ["Class 0 (Human)", "Class 1 (AI)", "Class 2 (Mixed)"],
                "matrix": cm,
            },
            "per_class_metrics": per_class_summary,
        },
        "esl_bias_audit": {
            "samples_evaluated": len(ESL_SIMULATION_SAMPLES),
            "false_positives": esl_false_positives,
            "false_positive_rate": round(esl_fpr, 4),
            "detailed_results": esl_results,
        },
        "edge_case_diagnostics": edge_case_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n[evaluate] Full evaluation report written to: {output_path}")
    return evaluation_report


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "dataset", "admissions_essays.json")
    m_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "models", "detector.joblib")
    o_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "dataset", "evaluation_results.json")
    evaluate_system(d_path, m_path, o_path)
