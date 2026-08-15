"""Orchestrate all measurements into a fixed-length feature vector.

The live API and the training script call the same function so a
highlighted sentence in the UI is using the same numbers the model saw.
"""

from __future__ import annotations

from features.base import EssayRecord, SentenceRecord, SentenceSpan
from features.burstiness import burstiness, nll_zscores
from features.embedding_coherence import embedding_coherence
from features.function_words import function_word_distance
from features.nlp_resources import get_embedder, get_gpt2, get_spacy, skip_neural
from features.perplexity import sentence_nlls
from features.pos_entropy import pos_entropy
from features.readability import readability_series
from features.segmentation import split_sentences
from features.sentence_length import length_stats
from features.transitions import transition_features
from features.vocabulary import window_ttr

# Order is part of the model contract. train.py and detect.py share this list.
FEATURE_NAMES: tuple[str, ...] = (
    "nll",
    "nll_z",
    "word_count",
    "word_count_z",
    "window_ttr",
    "transition_count",
    "transition_density",
    "embed_sim_prev",
    "embed_sim_next",
    "fk_grade",
    "fk_z",
    "pos_local_entropy",
    "essay_burstiness",
    "essay_len_mean",
    "essay_len_var",
    "essay_ttr",
    "essay_fw_cosine",
    "essay_pos_entropy",
    "essay_transition_density",
    "essay_embed_coherence_mean",
    "essay_fk_var",
)


def extract_essay(text: str, *, use_neural: bool | None = None) -> EssayRecord:
    if use_neural is None:
        use_neural = not skip_neural()

    nlp = get_spacy()
    sentences = split_sentences(text, nlp)
    if not sentences:
        return EssayRecord(
            text=text,
            sentences=[],
            essay_features={name: 0.0 for name in FEATURE_NAMES if name.startswith("essay_")},
            essay_explanations={},
            neural_used={"gpt2": False, "minilm": False, "spacy": nlp is not None},
        )

    model = tok = embedder = None
    if use_neural:
        model, tok = get_gpt2()
        embedder = get_embedder()

    nlls, nll_backend = sentence_nlls(text, sentences, model, tok)
    burst, burst_expl = burstiness(nlls)
    nll_z, nll_z_expl = nll_zscores(nlls)

    counts, len_mean, len_var, len_z, len_expl, len_z_expl = length_stats(sentences)
    ttrs, ttr_expl, essay_ttr = window_ttr(sentences)
    fw_dist, fw_expl = function_word_distance(text)
    pos_essay, pos_local, pos_essay_expl, pos_local_expl = pos_entropy(sentences, nlp)
    tr_counts, tr_dens, tr_expl, essay_tr, essay_tr_expl = transition_features(sentences)
    prev, nxt, mean_sim, emb_expl, emb_essay_expl, emb_backend = embedding_coherence(
        sentences, embedder
    )
    fk, fk_z, fk_var, fk_expl, fk_var_expl = readability_series(sentences)

    essay_features = {
        "essay_burstiness": burst,
        "essay_len_mean": len_mean,
        "essay_len_var": len_var,
        "essay_ttr": essay_ttr,
        "essay_fw_cosine": fw_dist,
        "essay_pos_entropy": pos_essay,
        "essay_transition_density": essay_tr,
        "essay_embed_coherence_mean": mean_sim,
        "essay_fk_var": fk_var,
    }
    essay_explanations = {
        "essay_burstiness": burst_expl,
        "essay_len_var": len_expl,
        "essay_fw_cosine": fw_expl,
        "essay_pos_entropy": pos_essay_expl,
        "essay_transition_density": essay_tr_expl,
        "essay_embed_coherence_mean": emb_essay_expl,
        "essay_fk_var": fk_var_expl,
    }

    records: list[SentenceRecord] = []
    for i, span in enumerate(sentences):
        feats = {
            "nll": nlls[i],
            "nll_z": nll_z[i],
            "word_count": float(counts[i]),
            "word_count_z": len_z[i],
            "window_ttr": ttrs[i],
            "transition_count": float(tr_counts[i]),
            "transition_density": tr_dens[i],
            "embed_sim_prev": prev[i],
            "embed_sim_next": nxt[i],
            "fk_grade": fk[i],
            "fk_z": fk_z[i],
            "pos_local_entropy": pos_local[i],
            **essay_features,
        }
        expl = {
            "nll_z": nll_z_expl[i],
            "word_count_z": len_z_expl[i],
            "window_ttr": ttr_expl[i],
            "transition_density": tr_expl[i],
            "embed_sim_prev": emb_expl[i],
            "fk_z": fk_expl[i],
            "pos_local_entropy": pos_local_expl[i],
            **essay_explanations,
        }
        records.append(SentenceRecord(span=span, features=feats, explanations=expl))

    return EssayRecord(
        text=text,
        sentences=records,
        essay_features=essay_features,
        essay_explanations=essay_explanations,
        neural_used={
            "gpt2": nll_backend == "distilgpt2",
            "minilm": emb_backend == "minilm",
            "spacy": nlp is not None,
            "nll_backend": nll_backend,
            "embed_backend": emb_backend,
        },
    )


def vectorize(record: SentenceRecord) -> list[float]:
    return [float(record.features.get(name, 0.0)) for name in FEATURE_NAMES]
