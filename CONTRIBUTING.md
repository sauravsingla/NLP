# Contributing

Contributions that improve correctness, reproducibility, teaching value, or coverage of real NLP problems are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check src tests examples
mypy src
pytest
```

## Contribution guidelines

- Keep each pull request focused on one problem.
- Add or update tests for behavior changes.
- Prefer small functions with typed inputs and predictable outputs.
- Keep model downloads and heavyweight libraries behind optional dependencies.
- Never commit private, licensed, or personally identifiable datasets.
- Report dataset source, license, split strategy, metrics, and random seed in experiments.
- Compare new deep-learning approaches against a simple baseline.

## Notebook checklist

A new notebook should include:

1. problem statement and expected learning outcome;
2. dataset source and license;
3. reproducible dependency installation;
4. deterministic seeds where supported;
5. train/validation/test separation;
6. suitable metrics, including class-level metrics for imbalanced data;
7. error analysis and limitations;
8. cleared large outputs and no embedded credentials.

## Commit style

Use short, natural commit messages that explain the change, for example:

- `Add named entity recognition example`
- `Fix data leakage in vectorizer training`
- `Document transformer inference requirements`
