"""Classical NLP baselines with leakage-safe scikit-learn pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

TextClassifier = Pipeline


@dataclass(frozen=True)
class ClassificationResult:
    """Summary metrics returned by :func:`evaluate_classifier`."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: dict[str, object]


def build_tfidf_classifier(
    *,
    max_features: int = 30_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    class_weight: str | dict[object, float] | None = "balanced",
    random_state: int = 42,
) -> TextClassifier:
    """Build a strong, interpretable text-classification baseline.

    The vectorizer lives inside the pipeline, preventing vocabulary and IDF
    leakage when the model is fitted only on training data.
    """
    if max_features < 1:
        raise ValueError("max_features must be positive")
    if min_df < 1:
        raise ValueError("min_df must be positive")
    if not 0 < max_df <= 1:
        raise ValueError("max_df must be in the interval (0, 1]")
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("ngram_range must contain increasing positive values")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model: TextClassifier,
    texts: Sequence[str],
    labels: Sequence[object],
) -> ClassificationResult:
    """Evaluate a fitted classifier using metrics suitable for imbalance."""
    if not texts:
        raise ValueError("texts must not be empty")
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    predictions = model.predict(texts)
    return ClassificationResult(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro")),
        weighted_f1=float(f1_score(labels, predictions, average="weighted")),
        report=classification_report(labels, predictions, output_dict=True, zero_division=0),
    )


def top_features(
    model: TextClassifier,
    *,
    class_label: object | None = None,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Return the most influential terms for a fitted logistic-regression class."""
    if limit < 1:
        raise ValueError("limit must be positive")

    vectorizer = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("classifier")
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(
        classifier, LogisticRegression
    ):
        raise TypeError("model must contain tfidf and logistic-regression steps")

    feature_names = vectorizer.get_feature_names_out()
    classes = list(classifier.classes_)
    coefficients = classifier.coef_

    if len(classes) == 2 and coefficients.shape[0] == 1:
        selected_class = classes[1] if class_label is None else class_label
        if selected_class not in classes:
            raise ValueError(f"unknown class_label: {selected_class!r}")
        weights = coefficients[0] if selected_class == classes[1] else -coefficients[0]
    else:
        selected_class = classes[0] if class_label is None else class_label
        if selected_class not in classes:
            raise ValueError(f"unknown class_label: {selected_class!r}")
        weights = coefficients[classes.index(selected_class)]

    top_indices: Iterable[int] = np.argsort(weights)[-limit:][::-1]
    return [(str(feature_names[index]), float(weights[index])) for index in top_indices]
