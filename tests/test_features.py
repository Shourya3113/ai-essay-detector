"""Unit tests for measured features. No chat model. No network."""

from __future__ import annotations

import os

os.environ["ESSAYTRACE_SKIP_NEURAL"] = "1"

from features.base import zscores
from features.burstiness import burstiness, nll_zscores
from features.char_lm import char_nll
from features.embedding_coherence import embedding_coherence
from features.extractor import FEATURE_NAMES, extract_essay, vectorize
from features.function_words import function_word_distance
from features.pos_entropy import coarse_pos, pos_entropy
from features.readability import count_syllables, flesch_kincaid_sentence
from features.segmentation import split_sentences
from features.sentence_length import length_stats, word_count
from features.transitions import MARKERS, transition_features
from features.vocabulary import type_token_ratio, window_ttr


HUMANISH = (
    "I burned the first tray of baklava on a Thursday that already smelled like bleach. "
    "My uncle swore in two languages and then handed me a cracked plastic scraper. "
    "Anyway, I kept the scraper in my backpack for a year. "
    "It made a stupid clicking sound against my chemistry notes."
)

AIISH = (
    "Moreover, this experience taught me the importance of resilience in today's world. "
    "Furthermore, I learned that leadership plays a pivotal role in community. "
    "In conclusion, it is important to note that growth is a tapestry of challenges. "
    "Additionally, I have always been committed to serving others with compassion."
)


def test_split_keeps_offsets():
    text = "Hello there. What now? Fine."
    spans = split_sentences(text)
    assert len(spans) == 3
    for s in spans:
        assert text[s.start : s.end] == s.text


def test_word_count_and_length_variance_lower_for_uniform():
    even = split_sentences(
        "This line has exactly seven words here. "
        "This line has exactly seven words here. "
        "This line has exactly seven words here."
    )
    varied = split_sentences(HUMANISH)
    _, _, var_even, _, _, _ = length_stats(even)
    _, _, var_var, _, _, _ = length_stats(varied)
    assert word_count(even[0].text) == 7
    assert var_even < var_var


def test_ttr_repeats_are_lower():
    assert type_token_ratio(["a", "a", "a", "a"]) < type_token_ratio(["a", "b", "c", "d"])
    spans = split_sentences(HUMANISH)
    vals, expl, essay = window_ttr(spans)
    assert len(vals) == len(spans)
    assert 0 < essay <= 1
    assert all(e for e in expl)


def test_function_word_distance_runs():
    dist, expl = function_word_distance(HUMANISH + " " + HUMANISH)
    assert 0 <= dist <= 2
    assert expl


def test_transition_density_flags_moreover():
    h = split_sentences(HUMANISH)
    a = split_sentences(AIISH)
    _, _, _, h_d, _ = transition_features(h)
    _, _, _, a_d, _ = transition_features(a)
    counts, _, _, _, _ = transition_features(a)
    assert a_d > h_d
    assert sum(counts) >= 3
    assert "moreover" in MARKERS


def test_pos_entropy_and_coarse_tagger():
    assert coarse_pos("the") == "DET"
    assert coarse_pos("running") == "VERB"
    spans = split_sentences(HUMANISH)
    essay_e, local, expl, local_e = pos_entropy(spans, nlp=None)
    assert essay_e >= 0
    assert len(local) == len(spans)
    assert expl and local_e


def test_readability_syllables_and_variance():
    assert count_syllables("cake") == 1
    assert count_syllables("education") >= 3
    easy = flesch_kincaid_sentence("The cat sat on the mat.")
    hard = flesch_kincaid_sentence(
        "Philosophical investigations necessitate considerable intellectual sophistication."
    )
    assert hard > easy


def test_burstiness_zscores_relative():
    nlls = [1.0, 1.02, 1.01, 4.5]
    var, expl = burstiness(nlls)
    zs, zexpl = nll_zscores(nlls)
    assert var > 0
    assert zs[-1] > 1.0
    assert zs[0] < 0
    assert expl and zexpl[-1]


def test_zscores_flat_list_zero():
    assert zscores([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_embedding_bow_smooth_vs_jump():
    smooth = split_sentences(
        "The community garden taught me patience and care. "
        "The community garden taught me patience every week. "
        "The community garden taught me care and patience."
    )
    jumpy = split_sentences(
        "The community garden taught me patience and care. "
        "The carburetor flooded at midnight on I-80. "
        "My sister collects bus transfers in a jam jar."
    )
    _, _, mean_s, _, _, _ = embedding_coherence(smooth)
    _, _, mean_j, _, _, _ = embedding_coherence(jumpy)
    assert mean_s > mean_j


def test_char_nll_positive():
    assert char_nll("Hello there, friend.") > 0


def test_extractor_feature_vector_contract():
    rec = extract_essay(HUMANISH, use_neural=False)
    assert rec.sentences
    for sent in rec.sentences:
        vec = vectorize(sent)
        assert len(vec) == len(FEATURE_NAMES)
        assert all(isinstance(x, float) for x in vec)
        assert sent.explanations
    rec2 = extract_essay(AIISH, use_neural=False)
    ai_tr = rec2.essay_features["essay_transition_density"]
    hu_tr = rec.essay_features["essay_transition_density"]
    assert ai_tr > hu_tr


def test_empty_essay():
    rec = extract_essay("   ", use_neural=False)
    assert rec.sentences == []
