from __future__ import annotations

import pytest

from nlp_reference import (
    build_tfidf_classifier,
    evaluate_classifier,
    normalize_text,
    top_features,
)


def test_normalize_text_handles_html_urls_and_email() -> None:
    text = "  Great&nbsp;Service! Contact Test@Example.com at https://example.com  "
    assert normalize_text(text) == "great service! contact [email] at"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(123)  # type: ignore[arg-type]


def test_classifier_trains_evaluates_and_explains() -> None:
    train_texts = [
        "excellent reliable product",
        "great quality and support",
        "happy with the purchase",
        "terrible broken product",
        "poor quality and delay",
        "unhappy with the purchase",
    ]
    train_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

    model = build_tfidf_classifier(ngram_range=(1, 1))
    model.fit(train_texts, train_labels)

    result = evaluate_classifier(model, train_texts, train_labels)
    features = top_features(model, class_label="positive", limit=3)

    assert result.accuracy >= 0.8
    assert result.macro_f1 >= 0.8
    assert len(features) == 3
    assert all(isinstance(term, str) and isinstance(weight, float) for term, weight in features)


def test_classifier_configuration_validation() -> None:
    with pytest.raises(ValueError, match="max_features"):
        build_tfidf_classifier(max_features=0)


def test_evaluation_requires_matching_lengths() -> None:
    model = build_tfidf_classifier()
    with pytest.raises(ValueError, match="same length"):
        evaluate_classifier(model, ["one", "two"], ["label"])
