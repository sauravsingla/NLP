# NLP: Classical to Modern Natural Language Processing

A practical reference repository for learning and applying natural language processing, from interpretable statistical baselines to transformer-based workflows.

The original notebooks are preserved as historical, hands-on examples. The reusable `src/nlp_reference` package adds tested implementations that can be used in scripts, services, experiments, and teaching material.

## What this repository covers

| Area | Traditional approach | Modern approach |
|---|---|---|
| Text preparation | regex cleaning, tokenization, stop-word handling | model-aware preprocessing |
| Text classification | TF-IDF + logistic regression | Hugging Face zero-shot classification |
| Semantic search | TF-IDF similarity | sentence-transformer embeddings |
| Topic discovery | LDA | BERTopic-ready workflow |
| Keyword extraction | frequency and TF-IDF | embedding-based ranking |
| Evaluation | accuracy, precision, recall, F1 | confidence-aware inference and shared metrics |

## Open dataset benchmarks

The benchmark runner supports four public datasets with different sizes and problem shapes:

- AG News: four-class news categorisation
- IMDb: long-form sentiment classification
- DBpedia 14: fourteen-class ontology classification
- Yelp Polarity: large-scale review sentiment classification

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[datasets,dev]"
pytest
python examples/benchmark_open_datasets.py --dataset all --max-train 10000 --max-test 2000
```

Each run uses deterministic sampling, fits only on the training split, records accuracy plus macro and weighted F1, and writes machine-readable results to `reports/benchmark_results.json`. See [docs/OPEN_DATASETS.md](docs/OPEN_DATASETS.md) for provenance and licensing guidance.

## Repository structure

```text
.
├── src/nlp_reference/       # reusable Python package
├── examples/                # runnable classical and modern examples
├── tests/                   # unit tests
├── docs/                    # problem and dataset guidance
├── .github/workflows/       # automated quality checks
├── *.ipynb                  # original educational notebooks
└── pyproject.toml           # package and tooling configuration
```

## Additional examples

```bash
python examples/classical_text_classification.py
pip install -e ".[transformers]"
python examples/modern_nlp_inference.py
```

## Design principles

- **Start with a strong baseline.** A small TF-IDF model is often faster, cheaper, and easier to explain than a large neural model.
- **Use one evaluation contract.** Traditional and modern systems should be compared with the same split and metrics.
- **Avoid data leakage.** Vectorizers and models are fitted only on training data through scikit-learn pipelines.
- **Keep inference reusable.** Public functions validate inputs and return predictable Python objects.
- **Make optional dependencies explicit.** Dataset and transformer libraries are installed only when needed.
- **Record provenance.** Benchmark outputs include source and license metadata.

## Included notebooks

The repository includes notebooks for topic modelling, document visualisation, and text classification. Some notebooks were created with older versions of Colab, spaCy, fastai, and related libraries. They remain useful for understanding the evolution of NLP, while the package and examples show the recommended current structure.

## Suggested learning path

1. Learn text cleaning and tokenization.
2. Build a TF-IDF classification baseline.
3. Run it on multiple public datasets.
4. Inspect model errors and class-level metrics.
5. Explore topic modelling and document similarity.
6. Compare a transformer or embedding model against the baseline.
7. Package the selected solution with tests and reproducible configuration.

## Quality checks

The project uses `pytest`, Ruff, MyPy, and GitHub Actions across supported Python versions.

## Responsible use

NLP models can reproduce bias, expose sensitive information, and fail on language varieties not represented in training data. Evaluate on data that reflects the deployment population, document limitations, protect personal data, and keep human review for high-impact decisions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull-request guidance.

## License

Released under the MIT License. See [LICENSE](LICENSE).
