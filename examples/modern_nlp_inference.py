"""Demonstrate modern NLP without task-specific training."""

from __future__ import annotations

from nlp_reference.modern import semantic_search, zero_shot_classify


def main() -> None:
    documents = [
        "A card payment was declined at a merchant.",
        "The mobile application is not opening after an update.",
        "A money transfer is pending and has not reached the beneficiary.",
        "The customer wants to change the registered phone number.",
    ]

    print("Semantic search")
    for result in semantic_search("recipient did not receive my transfer", documents, top_k=2):
        print(f"{result['rank']}. {result['score']:.3f} - {result['document']}")

    print("\nZero-shot classification")
    result = zero_shot_classify(
        "The transfer is still pending after two days.",
        ["card issue", "app issue", "money transfer", "profile update"],
    )
    print(result["labels"][0], result["scores"][0])


if __name__ == "__main__":
    main()
