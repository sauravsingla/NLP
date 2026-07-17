"""Reusable building blocks for classical and modern NLP workflows."""

from nlp_reference.classical import (
    ClassificationResult,
    build_tfidf_classifier,
    evaluate_classifier,
    top_features,
)
from nlp_reference.preprocessing import normalize_text

__all__ = [
    "ClassificationResult",
    "build_tfidf_classifier",
    "evaluate_classifier",
    "normalize_text",
    "top_features",
]
