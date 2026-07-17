# NLP Reference: Classical Models, Transformers and Reproducible Benchmarks

[![CI](https://github.com/sauravsingla/NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/NLP/actions/workflows/ci.yml)
[![Open dataset benchmarks](https://github.com/sauravsingla/NLP/actions/workflows/open-dataset-benchmarks.yml/badge.svg)](https://github.com/sauravsingla/NLP/actions/workflows/open-dataset-benchmarks.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A practical, tested NLP reference repository that connects traditional statistical methods with modern transformer and embedding workflows. It is designed for learners, instructors, data scientists and engineers who need examples that are understandable, reproducible and suitable for extension.

The original Colab notebooks are preserved as historical learning material. The recommended implementations live in the typed `src/nlp_reference` package, runnable examples, tests and automated benchmark workflows.

## Why this repository is useful

Many NLP repositories show a single notebook or one model on one dataset. This project instead provides a shared evaluation contract across multiple open datasets and models, while keeping the code lightweight enough to run on a CPU.

- **Traditional and modern approaches:** TF-IDF baselines, transformer inference and sentence embeddings.
- **Comparable public benchmarks:** the same splits, seeds, preprocessing rules and metrics.
- **Efficiency reporting:** model quality, fitting time, prediction time and throughput.
- **Reusable code:** typed modules rather than notebook-only implementations.
- **Reproducible engineering:** tests, linting, type checking and GitHub Actions.
- **Responsible documentation:** dataset provenance, licence guidance, limitations and security notes.

## Navigation

| Goal | Start here |
|---|---|
| Run the benchmark leaderboard | [Benchmark leaderboard](#benchmark-leaderboard) |
| Understand the codebase | [Architecture](docs/ARCHITECTURE.md) |
| Reproduce and report results | [Reproducibility guide](docs/REPRODUCIBILITY.md) |
| Select an NLP solution | [Problem selection guide](docs/NLP_PROBLEM_GUIDE.md) |
| Review dataset sources and licences | [Open datasets](docs/OPEN_DATASETS.md) |
| Contribute a model, dataset or notebook | [Contributing](CONTRIBUTING.md) |
| Report a security concern | [Security policy](SECURITY.md) |

## Coverage

| Problem area | Classical reference | Modern reference |
|---|---|---|
| Text preparation | regex normalisation and conservative cleaning | model-aware preprocessing |
| Text classification | TF-IDF with Logistic Regression, Linear SVM and Complement Naive Bayes | Hugging Face zero-shot inference |
| Semantic search | sparse vector similarity | sentence-transformer embeddings |
| Topic discovery | LDA notebooks | BERTopic-ready workflow direction |
| Document visualisation | dimensionality-reduction notebooks | embedding-based visualisation direction |
| Evaluation | accuracy, macro F1 and weighted F1 | shared metrics plus confidence-aware inference |
| Efficiency | fitting and prediction timing | model-specific latency and resource analysis direction |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Run the small classical example:

```bash
python examples/classical_text_classification.py
```

Install optional transformer dependencies and run modern inference:

```bash
pip install -e ".[transformers]"
python examples/modern_nlp_inference.py
```

## Benchmark leaderboard

The leaderboard evaluates three efficient classical models across four public text-classification datasets.

### Models

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Complement Naive Bayes

### Datasets

| Dataset | Task | Classes | What it tests |
|---|---|---:|---|
| AG News | news categorisation | 4 | short text and topic separation |
| IMDb | review sentiment | 2 | long-form sentiment and sparse features |
| DBpedia 14 | ontology classification | 14 | larger multiclass discrimination |
| Yelp Polarity | review sentiment | 2 | scale and domain-specific sentiment |

Install dataset support and generate the leaderboard:

```bash
pip install -e ".[datasets,dev]"
python examples/generate_leaderboard.py \
  --dataset all \
  --model all \
  --max-train 10000 \
  --max-test 2000 \
  --seed 42
```

Generated outputs:

```text
reports/leaderboard.json     machine-readable results
reports/LEADERBOARD.md       ranked Markdown leaderboard
```

Each entry records:

- accuracy;
- macro F1;
- weighted F1;
- fitting time;
- prediction time;
- inference examples per second;
- train and test sample counts.

Macro F1 is the primary ranking metric because it gives equal importance to every class. Accuracy remains useful for balanced datasets, while weighted F1 reflects performance according to class frequency. Runtime measurements depend on hardware and should only be compared when the execution environment is equivalent.

GitHub Actions runs the benchmark in a clean environment, displays the Markdown leaderboard in the workflow summary and uploads both report files as an artifact. A successful workflow is the source of truth for externally executed results; do not treat placeholder or smoke-test values as published benchmark claims.

## Repository structure

```text
.
├── src/nlp_reference/
│   ├── preprocessing.py      safe text normalisation
│   ├── classical.py          TF-IDF pipelines and evaluation
│   ├── datasets.py           public-dataset registry
│   ├── leaderboard.py        model comparison and reporting
│   └── modern.py             optional transformers and embeddings
├── examples/                 runnable end-to-end scripts
├── tests/                    unit tests
├── docs/                     architecture, datasets and guidance
├── reports/                  generated benchmark outputs
├── .github/workflows/        CI and public benchmark automation
├── *.ipynb                   historical educational notebooks
├── pyproject.toml            packaging and tool configuration
└── LICENSE                   MIT licence
```

## Python API example

```python
from nlp_reference.classical import build_tfidf_classifier, evaluate_classifier

train_texts = [
    "The payment was completed successfully",
    "The account transfer failed unexpectedly",
    "The service was quick and reliable",
    "The application repeatedly crashed",
]
train_labels = ["positive", "negative", "positive", "negative"]

model = build_tfidf_classifier(random_state=42)
model.fit(train_texts, train_labels)

result = evaluate_classifier(
    model,
    ["The transfer worked immediately", "The application failed again"],
    ["positive", "negative"],
)
print(result.macro_f1)
```

## Evaluation principles

1. **Begin with a strong baseline.** A small interpretable model often provides an excellent quality, cost and latency reference.
2. **Prevent leakage.** Vectorisers and models are fitted only on training data through scikit-learn pipelines.
3. **Use the official splits.** Dataset train and test partitions are not recombined.
4. **Make sampling deterministic.** Sample limits use a recorded seed.
5. **Compare with one contract.** Models use equivalent datasets, limits and metrics.
6. **Measure quality and efficiency.** Accuracy alone is not enough for production decisions.
7. **Report limitations honestly.** Sampled runs are not presented as state-of-the-art full-dataset results.

## Learning paths

### Beginner

1. Review text cleaning and tokenisation.
2. Run `examples/classical_text_classification.py`.
3. Inspect TF-IDF features and class-level metrics.
4. Explore the historical topic-modelling notebooks.

### Practitioner

1. Generate the multi-dataset leaderboard.
2. Compare macro F1 with throughput and fitting time.
3. Add error analysis for the selected dataset.
4. Package the chosen model behind a service or batch pipeline.

### Advanced

1. Add a transformer fine-tuning benchmark track.
2. Compare dense and sparse retrieval.
3. Extend evaluation to multilingual and imbalanced datasets.
4. Add calibration, robustness and bias analysis.
5. Reproduce results on different hardware and publish the environment details.

## Historical notebooks

The repository contains notebooks created with earlier Colab, spaCy, fastai and related APIs. They are retained to show the evolution of NLP workflows, but they are not the recommended production interface. New work should use the package and examples, or modernise a notebook with explicit versions, dataset provenance, tests and documented limitations.

## Quality checks

```bash
ruff check src tests examples
mypy src
pytest
```

The automated checks validate supported Python environments and the benchmark workflow validates public-dataset execution separately.

## Responsible NLP

NLP systems can reproduce social bias, expose sensitive information, perform poorly on underrepresented language varieties and become unreliable under distribution shift. Evaluate on data representative of the intended population, protect personal data, document failure modes and retain human oversight for consequential decisions.

Third-party datasets and models have their own licences and usage terms. Review them before commercial or production use.

## Contributing

Contributions are welcome for correctness, tests, datasets, multilingual coverage, modern models, retrieval, explainability and deployment examples. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

A benchmark contribution should include its source, licence, exact split, seed, model configuration, quality metrics, runtime context and known limitations.

## Security

Do not commit credentials, private data or proprietary model artefacts. See [SECURITY.md](SECURITY.md) for responsible-disclosure and production-hardening guidance.

## Citation

When this repository supports teaching, research or an implementation, cite the repository URL and the exact commit used so that others can reproduce the same version.

```text
Saurav Singla. NLP Reference: Classical Models, Transformers and
Reproducible Benchmarks. GitHub repository, accessed at a specific commit.
```

## Licence

Released under the [MIT License](LICENSE). Dataset and model licences remain governed by their respective providers.