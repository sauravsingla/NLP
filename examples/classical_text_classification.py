"""Train and inspect a small TF-IDF text classifier."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from nlp_reference import build_tfidf_classifier, evaluate_classifier, top_features


def main() -> None:
    texts = [
        "The payment was fast and the app was easy to use",
        "Excellent support resolved my problem quickly",
        "The transfer completed immediately",
        "The interface is clear and reliable",
        "The payment failed and support did not respond",
        "The app crashes whenever I open the history",
        "The transfer was delayed for several days",
        "The interface is confusing and very slow",
    ]
    labels = ["positive"] * 4 + ["negative"] * 4

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = build_tfidf_classifier(ngram_range=(1, 2))
    model.fit(train_texts, train_labels)
    result = evaluate_classifier(model, test_texts, test_labels)

    print(f"accuracy={result.accuracy:.3f}")
    print(f"macro_f1={result.macro_f1:.3f}")
    print("top positive features:")
    for term, weight in top_features(model, class_label="positive", limit=5):
        print(f"  {term:<20} {weight:.3f}")


if __name__ == "__main__":
    main()
