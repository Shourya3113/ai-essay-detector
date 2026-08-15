"""Measured linguistic features for sentence-level AI-writing detection.

Nothing in this package asks a chat model for a verdict. Features are
counts, distances, entropies, and local-base-model log-probabilities.
"""

from features.extractor import extract_essay, FEATURE_NAMES

__all__ = ["extract_essay", "FEATURE_NAMES"]
