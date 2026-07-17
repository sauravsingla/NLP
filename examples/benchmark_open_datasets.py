"""Benchmark the classical NLP baseline on public datasets.

Examples:
    python examples/benchmark_open_datasets.py --dataset ag_news
    python examples/benchmark_open_datasets.py --dataset all --max-train 10000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nlp_reference.classical import build_tfidf_classifier, evaluate_classifier
from nlp_reference.datasets import available_datasets, load_text_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *available_datasets()),
        help="Public dataset to run, or all supported datasets.",
    )
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--max-test", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.json"))
    return parser.parse_args()


def run_benchmark(name: str, args: argparse.Namespace) -> dict[str, object]:
    dataset = load_text_dataset(
        name,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )
    model = build_tfidf_classifier(random_state=args.seed)
    model.fit(dataset.train_texts, dataset.train_labels)
    result = evaluate_classifier(model, dataset.test_texts, dataset.test_labels)
    return {
        "dataset": dataset.name,
        "source": dataset.source,
        "license": dataset.license_name,
        "train_examples": len(dataset.train_texts),
        "test_examples": len(dataset.test_texts),
        "metrics": asdict(result),
    }


def main() -> None:
    args = parse_args()
    names = available_datasets() if args.dataset == "all" else (args.dataset,)
    results = [run_benchmark(name, args) for name in names]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for item in results:
        metrics = item["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{item['dataset']}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    print(f"Saved reproducible results to {args.output}")


if __name__ == "__main__":
    main()
