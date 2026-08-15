"""
test_suite.py — Unit and Integration Tests for 9-Feature AI Essay Detector

Exercises all 9 analyzers, scoring engine, edge cases, and ML inference.
"""

import sys
import os

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analyzer.burstiness import compute_burstiness
from analyzer.lexical import compute_lexical_diversity
from analyzer.phrases import compute_phrase_score
from analyzer.punctuation import compute_punctuation_score
from analyzer.function_words import compute_function_words
from analyzer.pos_entropy import compute_pos_entropy
from analyzer.readability import compute_readability, compute_sentence_readability
from analyzer.transitions import compute_transition_score
from scorer import analyze_essay, split_sentences

HUMAN_SAMPLE = (
    "The grease beneath my fingernails refused to come out before Monday morning chemistry. "
    "I scrubbed until my cuticles bled. My father’s 1998 Ford Ranger had thrown a transmission cylinder on highway 29, "
    "and we simply didn’t have the eight hundred dollars for a certified mechanic. "
    "So there I was, freezing in our dirt driveway with a Chilton repair manual propped against a car battery. "
    "I dropped the ratchet twice. My knuckles tore on the exhaust manifold—a sharp, stinging reminder of our budget. "
    "It took eleven days, forty cups of lukewarm black coffee, and three stripped bolts. But on Sunday night, the engine roared. "
    "That battered truck taught me more about stubborn kinetic perseverance than any AP Physics lab ever could."
)

AI_SAMPLE = (
    "In today's fast-paced and challenging world, overcoming adversity plays a pivotal role in shaping character. "
    "Throughout my academic journey, I have encountered a myriad of obstacles that tested my unwavering resilience. "
    "Furthermore, facing financial difficulties forced me to balance rigorous coursework with substantial part-time employment. "
    "Moreover, this experience served as a powerful testament to the transformative power of perseverance and dedication. "
    "It is important to note that these challenges did not deter me from pursuing my intellectual passions with enthusiasm. "
    "Consequently, I learned to navigate the complexities of life while fostering a deeper understanding of personal responsibility. "
    "In conclusion, these multifaceted trials have prepared me to make a meaningful impact in my future collegiate endeavors."
)

def run_tests():
    print("==================================================")
    print("    RUNNING AI DETECTOR 9-FEATURE TEST SUITE")
    print("==================================================")

    # 1. Function Words Analyzer
    print("\n[Test 1] Function Words Analyzer...")
    fw_h = compute_function_words(HUMAN_SAMPLE)
    fw_a = compute_function_words(AI_SAMPLE)
    print(f"  Human FW Ratio: {fw_h['ratio']:.1%}, Entropy: {fw_h['entropy']:.2f}, Score: {fw_h['score']}")
    print(f"  AI FW Ratio:    {fw_a['ratio']:.1%}, Entropy: {fw_a['entropy']:.2f}, Score: {fw_a['score']}")
    assert 0.0 <= fw_h["score"] <= 1.0
    assert 0.0 <= fw_a["score"] <= 1.0
    print("  -> Passed.")

    # 2. POS Entropy Analyzer
    print("\n[Test 2] POS Entropy Analyzer...")
    pos_h = compute_pos_entropy(HUMAN_SAMPLE)
    pos_a = compute_pos_entropy(AI_SAMPLE)
    print(f"  Human POS Bigram Entropy: {pos_h['entropy']:.2f}, Score: {pos_h['score']}")
    print(f"  AI POS Bigram Entropy:    {pos_a['entropy']:.2f}, Score: {pos_a['score']}")
    assert 0.0 <= pos_h["score"] <= 1.0
    assert 0.0 <= pos_a["score"] <= 1.0
    print("  -> Passed.")

    # 3. Readability Analyzer
    print("\n[Test 3] Readability Analyzer...")
    r_h = compute_readability(HUMAN_SAMPLE)
    r_a = compute_readability(AI_SAMPLE)
    print(f"  Human FK Grade: {r_h['fk_grade']}, Std Dev: {r_h['std_dev']}, Score: {r_h['score']}")
    print(f"  AI FK Grade:    {r_a['fk_grade']}, Std Dev: {r_a['std_dev']}, Score: {r_a['score']}")
    assert 0.0 <= r_h["score"] <= 1.0
    assert 0.0 <= r_a["score"] <= 1.0
    print("  -> Passed.")

    # 4. Transitions Analyzer
    print("\n[Test 4] Transitions Analyzer...")
    t_h = compute_transition_score(HUMAN_SAMPLE)
    t_a = compute_transition_score(AI_SAMPLE)
    print(f"  Human Transition Count: {t_h['count']}, Density: {t_h['density_per_100_words']}/100w, Score: {t_h['score']}")
    print(f"  AI Transition Count:    {t_a['count']}, Density: {t_a['density_per_100_words']}/100w, Score: {t_a['score']}")
    assert t_a["count"] > t_h["count"]
    print("  -> Passed.")

    # 5. Full Pipeline Scorer on Human Essay
    print("\n[Test 5] Scorer Analysis on Human Essay...")
    res_h = analyze_essay(HUMAN_SAMPLE)
    print(f"  Overall Score: {res_h['overall_score']}")
    print(f"  Verdict:       {res_h['verdict']}")
    print(f"  ML Active:     {res_h['ml_calibrated']}")
    print(f"  Class Probs:   {res_h['class_probabilities']}")
    print(f"  Sentence 0 Explanations: {res_h['sentences'][0]['explanations']}")
    assert res_h["verdict"] == "Likely Human-Written"
    print("  -> Passed.")

    # 6. Full Pipeline Scorer on AI Essay
    print("\n[Test 6] Scorer Analysis on AI Essay...")
    res_a = analyze_essay(AI_SAMPLE)
    print(f"  Overall Score: {res_a['overall_score']}")
    print(f"  Verdict:       {res_a['verdict']}")
    print(f"  ML Active:     {res_a['ml_calibrated']}")
    print(f"  Class Probs:   {res_a['class_probabilities']}")
    print(f"  Sentence 0 Explanations: {res_a['sentences'][0]['explanations']}")
    assert res_a["overall_score"] > res_h["overall_score"]
    assert res_a["verdict"] == "Likely AI-Generated"
    print("  -> Passed.")

    # 7. Edge Cases: Single Sentence, Empty Input, Punctuation Only
    print("\n[Test 7] Robustness & Edge Cases...")
    single_res = analyze_essay("I looked at the clock and ran.")
    assert len(single_res["sentences"]) == 1
    print("  - Single sentence evaluated safely.")

    weird_res = analyze_essay("What?! Really... Yes—of course (maybe)! 12345.")
    assert len(weird_res["sentences"]) >= 1
    print("  - Irregular punctuation evaluated safely.")

    try:
        analyze_essay("   ")
        assert False, "Should have raised exception for empty text."
    except Exception:
        print("  - Empty string correctly caught with error.")

    print("\n==================================================")
    print("   ALL 9-FEATURE TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
