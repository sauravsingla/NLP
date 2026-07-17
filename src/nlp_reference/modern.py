"""Modern NLP helpers with lazy optional dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def semantic_search(
    query: str,
    documents: Sequence[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Rank documents by cosine similarity using sentence embeddings.

    Install the optional dependencies with ``pip install -e '.[transformers]'``.
    The model is loaded inside the function so importing the core package stays
    lightweight.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not documents:
        raise ValueError("documents must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "semantic_search requires the 'transformers' optional dependencies"
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [query, *documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = embeddings[1:] @ embeddings[0]
    ranked_indices = np.argsort(scores)[::-1][: min(top_k, len(documents))]

    return [
        {
            "rank": rank,
            "document": documents[int(index)],
            "score": float(scores[int(index)]),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def zero_shot_classify(
    text: str,
    candidate_labels: Sequence[str],
    *,
    model_name: str = "facebook/bart-large-mnli",
    multi_label: bool = False,
) -> dict[str, Any]:
    """Classify text against labels without task-specific model training."""
    if not text.strip():
        raise ValueError("text must not be empty")
    if not candidate_labels or any(not label.strip() for label in candidate_labels):
        raise ValueError("candidate_labels must contain non-empty strings")

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "zero_shot_classify requires the 'transformers' optional dependencies"
        ) from exc

    classifier = pipeline("zero-shot-classification", model=model_name)
    result = classifier(text, list(candidate_labels), multi_label=multi_label)
    return dict(result)
