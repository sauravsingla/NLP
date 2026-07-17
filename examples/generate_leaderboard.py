"""Generate JSON and Markdown leaderboards on supported public datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlp_reference.datasets import available_datasets, load_text_dataset
from nlp_reference.leaderboard import benchmark_model, model_registry, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *available_datasets()),
        help="Dataset to benchmark, or all supported datasets.",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=("all", *model_registry().keys()),
        help="Model to benchmark, or all registered models.",
    )
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--max-test", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-output", type=Path, default=Path("reports/leaderboard.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/LEADERBOARD.md")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = available_datasets() if args.dataset == "all" else (args.dataset,)
    registry = model_registry()
    models = tuple(registry) if args.model == "all" else (args.model,)
    results = []

    for dataset_name in datasets:
        dataset = load_text_dataset(
            dataset_name,
            max_train=args.max_train,
            max_test=args.max_test,
            seed=args.seed,
        )
        for model_name in models:
            print(f"Running {model_name} on {dataset.name}...")
            result = benchmark_model(
                dataset_name=dataset.name,
                model_name=model_name,
                model=registry[model_name](args.seed),
                train_texts=dataset.train_texts,
                train_labels=dataset.train_labels,
                test_texts=dataset.test_texts,
                test_labels=dataset.test_labels,
            )
            results.append(result)
            print(
                f"  accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}, "
                f"fit={result.fit_seconds:.2f}s"
            )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(results), encoding="utf-8")
    print(f"Saved machine-readable results to {args.json_output}")
    print(f"Saved Markdown leaderboard to {args.markdown_output}")


if __name__ == "__main__":
    main()
