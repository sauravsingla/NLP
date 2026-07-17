# Contributing

Contributions that improve correctness, reproducibility, teaching value or coverage of real NLP problems are welcome.

## Before starting

Use an issue or focused proposal for substantial additions such as a new benchmark track, dataset family, deployment stack or repository-wide refactor. Small documentation fixes and isolated tests can be submitted directly.

A contribution should answer three questions:

1. What NLP problem does this solve?
2. Why is the proposed approach useful compared with the existing baseline?
3. How can another person reproduce and evaluate it?

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install optional groups only when required:

```bash
pip install -e ".[datasets]"
pip install -e ".[transformers]"
```

## Required checks

Before opening a pull request, run:

```bash
ruff check src tests examples
mypy src
pytest
```

Changes affecting public datasets or benchmark code should also run an appropriate smoke benchmark with small sample limits.

## Pull-request scope

- Keep each pull request focused on one coherent change.
- Explain the problem, design choice, validation and limitations.
- Add or update tests for behavioural changes.
- Update the README or relevant guide when the public interface changes.
- Avoid unrelated formatting or notebook-output changes.
- Do not commit generated caches, model weights or large datasets.

## Code standards

- Prefer small functions with typed inputs and predictable outputs.
- Validate public-function inputs and provide useful errors.
- Keep preprocessing and model fitting inside leakage-safe pipelines.
- Put heavyweight libraries behind optional dependencies.
- Preserve deterministic seeds where the upstream library supports them.
- Avoid hidden network calls in core utility functions.
- Keep examples runnable from the repository root.

## Adding a dataset

A dataset contribution must document:

- canonical name and upstream source;
- task and label meaning;
- licence or usage terms;
- official train, validation and test splits;
- text and label fields;
- any transformations or filtering;
- expected class balance and known limitations.

Register it through the common dataset interface and add tests for registry discovery, split handling and deterministic sampling. Never commit private, licensed-without-redistribution or personally identifiable data.

## Adding a leaderboard model

A new model should:

1. use the same train and test examples as existing entries;
2. avoid test-set tuning and preprocessing leakage;
3. expose a stable registry name;
4. report accuracy, macro F1 and weighted F1;
5. record fitting and prediction time;
6. document important hyperparameters;
7. include tests that run without downloading a large model unless it belongs to an optional benchmark track.

Deep-learning and LLM approaches should be compared with a simple baseline. Report hardware, precision, batch size, model revision and approximate resource requirements.

## Benchmark-result policy

Do not manually enter unverified leaderboard scores as though they came from the automated workflow. A published result should identify:

```text
Dataset and revision
Train/test sample counts
Model and hyperparameters
Seed
Dependency versions
Hardware
Quality metrics
Runtime metrics
Known limitations
Exact command or workflow run
```

Sampled development runs must be labelled as sampled. Runtime results from different hardware should not be ranked as directly comparable.

## Notebook checklist

A new or modernised notebook should include:

1. problem statement and learning outcome;
2. dataset source and licence;
3. reproducible dependency installation;
4. deterministic seeds where supported;
5. train/validation/test separation;
6. suitable metrics and class-level analysis;
7. baseline comparison;
8. error analysis and limitations;
9. cleared large outputs;
10. no credentials, private data or local-only file paths.

## Documentation style

- Write for readers who may be new to the specific technique.
- Explain why a method is used, not only how to call it.
- Distinguish historical notebooks from recommended current code.
- Use complete commands that work from the repository root.
- Avoid performance or state-of-the-art claims without reproducible evidence.

## Commit style

Use short, natural commit messages that explain one change, for example:

- `Add named entity recognition example`
- `Fix data leakage in vectorizer training`
- `Document transformer inference requirements`
- `Benchmark linear SVM on AG News`

## Security and responsible use

Do not include secrets, unsafe deserialisation patterns or private data. Review [SECURITY.md](SECURITY.md) and document material bias, privacy, misuse or reliability risks introduced by a new model or dataset.