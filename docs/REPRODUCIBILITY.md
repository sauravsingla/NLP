# Reproducibility Guide

This guide defines how to reproduce, interpret and report results from this repository.

## Recommended environment

- Python 3.10 or 3.11
- a fresh virtual environment
- pinned project dependency ranges from `pyproject.toml`
- CPU execution for the classical benchmark track

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[datasets,dev]"
```

## Verify the installation

```bash
ruff check src tests examples
mypy src
pytest
```

## Reproduce the leaderboard

```bash
python examples/generate_leaderboard.py \
  --dataset all \
  --model all \
  --max-train 10000 \
  --max-test 2000 \
  --seed 42
```

The command writes:

- `reports/leaderboard.json`, suitable for automated analysis;
- `reports/LEADERBOARD.md`, suitable for GitHub and human review.

## Full-dataset runs

Sample limits keep routine runs affordable. For a full-dataset benchmark, omit the limits only when the runner and dataset loader support unbounded values, or set limits equal to the complete split sizes. Record the exact command, dependency versions and hardware used.

## What can be compared directly

Quality metrics are comparable when runs use the same:

- dataset revision;
- official split;
- sample limits;
- seed;
- preprocessing contract;
- model configuration;
- dependency versions.

Runtime and throughput are not directly comparable across different machines. Report CPU model, core count, memory, operating system and whether the dataset cache was warm.

## Required experiment record

A publishable result should include:

```text
Dataset and revision:
Dataset licence:
Train/test sample counts:
Model and hyperparameters:
Random seed:
Python and package versions:
Hardware:
Accuracy:
Macro F1:
Weighted F1:
Fit time:
Prediction time:
Examples per second:
Known limitations:
```

## Determinism

The repository passes a seed to supported sampling and model operations. Complete bit-for-bit determinism can still vary by operating system, numerical library and hardware. Small numerical differences should be reported rather than hidden.

## Data leakage controls

TF-IDF is fitted inside the training pipeline. Test text must never influence vocabulary construction, inverse-document-frequency statistics, feature selection, threshold selection or hyperparameter tuning.

## GitHub Actions

The benchmark workflow runs in a clean hosted environment, produces a workflow summary and uploads generated reports as artifacts. CI results are useful for reproducibility checks, but hosted-runner hardware can change; therefore runtime figures should not be treated as permanent performance claims.

## Responsible reporting

Do not present sampled benchmark scores as full-dataset state-of-the-art results. Clearly distinguish smoke tests, development runs, full evaluations and externally reproduced results.