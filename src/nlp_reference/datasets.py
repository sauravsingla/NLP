"""Small, reproducible loaders for public NLP benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextDataset:
    """Normalized train and test splits for text classification."""

    name: str
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]
    source: str
    license_name: str


_DATASETS: dict[str, dict[str, Any]] = {
    "ag_news": {
        "text": "text",
        "label": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "source": "Hugging Face datasets/ag_news",
        "license": "Dataset card terms",
    },
    "imdb": {
        "text": "text",
        "label": "label",
        "labels": ["negative", "positive"],
        "source": "Stanford Large Movie Review Dataset via Hugging Face",
        "license": "Dataset authors' terms",
    },
    "dbpedia_14": {
        "text": "content",
        "label": "label",
        "labels": [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ],
        "source": "DBpedia ontology classification via Hugging Face",
        "license": "CC BY-SA 3.0 / GFDL-derived data",
    },
    "yelp_polarity": {
        "text": "text",
        "label": "label",
        "labels": ["negative", "positive"],
        "source": "Yelp Review Polarity via Hugging Face",
        "license": "Dataset card terms",
    },
}


def available_datasets() -> tuple[str, ...]:
    """Return supported public dataset identifiers."""
    return tuple(sorted(_DATASETS))


def load_text_dataset(
    name: str,
    *,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 42,
) -> TextDataset:
    """Download and normalize one public dataset using ``datasets``.

    Sampling is deterministic and happens after shuffling, so quick benchmark
    runs remain representative while full runs can omit the limits.
    """
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {available_datasets()}")
    if max_train is not None and max_train < 1:
        raise ValueError("max_train must be positive")
    if max_test is not None and max_test < 1:
        raise ValueError("max_test must be positive")

    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise ImportError(
            'Install dataset support with: pip install -e ".[datasets]"'
        ) from exc

    config = _DATASETS[name]
    loaded = load_dataset(name)
    train = _sample(loaded["train"], max_train, seed)
    test_split = "test" if "test" in loaded else "validation"
    test = _sample(loaded[test_split], max_test, seed + 1)

    text_column = str(config["text"])
    label_column = str(config["label"])
    return TextDataset(
        name=name,
        train_texts=[str(value) for value in train[text_column]],
        train_labels=[int(value) for value in train[label_column]],
        test_texts=[str(value) for value in test[text_column]],
        test_labels=[int(value) for value in test[label_column]],
        label_names=list(config["labels"]),
        source=str(config["source"]),
        license_name=str(config["license"]),
    )


def _sample(dataset: Any, limit: int | None, seed: int) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))
