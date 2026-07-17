"""Model registry, timing helpers, and Markdown leaderboard rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Sequence

from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

ModelFactory = Callable[[int], Pipeline]


@dataclass(frozen=True)
class LeaderboardResult:
    """One reproducible model-and-dataset benchmark result."""

    dataset: str
    model: str
    train_examples: int
    test_examples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    fit_seconds: float
    predict_seconds: float
    examples_per_second: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _pipeline(classifier: ClassifierMixin) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    max_features=30_000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def model_registry() -> dict[str, ModelFactory]:
    """Return efficient, strong classical NLP models with a common interface."""
    return {
        "logistic_regression": lambda seed: _pipeline(
            LogisticRegression(
                class_weight="balanced", max_iter=1_000, random_state=seed
            )
        ),
        "linear_svm": lambda seed: _pipeline(
            SGDClassifier(
                loss="hinge",
                class_weight="balanced",
                max_iter=2_000,
                tol=1e-3,
                random_state=seed,
            )
        ),
        "complement_naive_bayes": lambda seed: _pipeline(ComplementNB(alpha=0.5)),
    }


def benchmark_model(
    *,
    dataset_name: str,
    model_name: str,
    model: Pipeline,
    train_texts: Sequence[str],
    train_labels: Sequence[object],
    test_texts: Sequence[str],
    test_labels: Sequence[object],
) -> LeaderboardResult:
    """Fit, time, and evaluate one model without leaking test information."""
    if not train_texts or not test_texts:
        raise ValueError("training and test data must not be empty")
    if len(train_texts) != len(train_labels) or len(test_texts) != len(test_labels):
        raise ValueError("text and label lengths must match")

    fit_start = perf_counter()
    model.fit(train_texts, train_labels)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    predictions = model.predict(test_texts)
    predict_seconds = perf_counter() - predict_start
    throughput = len(test_texts) / max(predict_seconds, 1e-12)

    return LeaderboardResult(
        dataset=dataset_name,
        model=model_name,
        train_examples=len(train_texts),
        test_examples=len(test_texts),
        accuracy=float(accuracy_score(test_labels, predictions)),
        macro_f1=float(f1_score(test_labels, predictions, average="macro")),
        weighted_f1=float(f1_score(test_labels, predictions, average="weighted")),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        examples_per_second=throughput,
    )


def render_markdown(results: Sequence[LeaderboardResult]) -> str:
    """Render results sorted by dataset and descending macro F1."""
    header = (
        "# NLP Benchmark Leaderboard\n\n"
        "Results are generated from official public dataset splits using identical "
        "sampling and evaluation rules. Runtime values depend on the runner hardware.\n\n"
        "| Dataset | Model | Accuracy | Macro F1 | Weighted F1 | Fit (s) | "
        "Predict (s) | Examples/s |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for item in sorted(results, key=lambda value: (value.dataset, -value.macro_f1)):
        rows.append(
            f"| {item.dataset} | {item.model} | {item.accuracy:.4f} | "
            f"{item.macro_f1:.4f} | {item.weighted_f1:.4f} | "
            f"{item.fit_seconds:.3f} | {item.predict_seconds:.3f} | "
            f"{item.examples_per_second:.1f} |"
        )
    return header + "\n".join(rows) + "\n"
