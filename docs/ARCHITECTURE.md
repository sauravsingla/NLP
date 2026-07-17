# Architecture

This repository separates educational notebooks from reusable, tested NLP components.

## System overview

```text
Open datasets
    |
    v
Dataset registry and deterministic sampling
    |
    v
Shared preprocessing and train/test contract
    |
    +----------------------+----------------------+
    |                      |                      |
    v                      v                      v
Logistic Regression     Linear SVM       Complement Naive Bayes
    |                      |                      |
    +----------------------+----------------------+
                           |
                           v
             Accuracy, F1 and runtime metrics
                           |
                           v
             JSON results and Markdown leaderboard
```

## Main components

### `src/nlp_reference/preprocessing.py`

Provides conservative text-normalisation utilities. Preprocessing is intentionally small because aggressive cleaning can remove information that modern tokenisers and classification models use.

### `src/nlp_reference/classical.py`

Contains leakage-safe TF-IDF classification pipelines and shared evaluation helpers. Vectorisation is kept inside each scikit-learn pipeline so vocabulary and inverse-document-frequency statistics are learned only from training data.

### `src/nlp_reference/datasets.py`

Defines the public-dataset registry, source metadata, licence notes and deterministic sampling rules. Dataset-specific field differences are converted into one common text-classification contract.

### `src/nlp_reference/leaderboard.py`

Builds comparable baseline models, measures fitting and prediction time, ranks results and renders machine-readable and human-readable reports.

### `src/nlp_reference/modern.py`

Keeps optional transformer and sentence-embedding workflows separate from the lightweight core package. This avoids forcing large model downloads on users who only need classical baselines.

## Benchmark contract

Every leaderboard entry follows the same rules:

1. use the official dataset train and test splits;
2. apply deterministic subsampling when limits are requested;
3. fit preprocessing and model parameters only on training examples;
4. evaluate on untouched test examples;
5. report accuracy, macro F1 and weighted F1;
6. record fitting time, prediction time and throughput;
7. preserve dataset source and licence metadata.

## Extension points

A new dataset should be registered in `datasets.py` and return the shared dataset object. A new model should be exposed through the model registry in `leaderboard.py`. Both additions should include tests and documentation before being included in the public leaderboard.

## Design trade-offs

The reference leaderboard prioritises reproducibility, interpretability and CPU accessibility. It is not intended to claim state-of-the-art results. Transformer fine-tuning, multilingual evaluation and retrieval workflows can be added as optional benchmark tracks without weakening the lightweight baseline track.