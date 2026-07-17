# Open-source dataset benchmark matrix

The reusable examples use public datasets through the Hugging Face `datasets` library. Data is downloaded at runtime and is not copied into this repository.

| Dataset | Problem | Classes | Recommended use |
|---|---|---:|---|
| AG News | news classification | 4 | fast multiclass smoke test |
| IMDb | sentiment classification | 2 | long-form document sentiment |
| DBpedia 14 | ontology classification | 14 | larger multiclass benchmark |
| Yelp Polarity | review sentiment | 2 | high-volume binary classification |

Run every classical benchmark with deterministic sampling:

```bash
pip install -e ".[datasets]"
python examples/benchmark_open_datasets.py --dataset all --max-train 10000 --max-test 2000
```

Run the full official splits by passing no limits after editing the command defaults, or call `load_text_dataset` directly with `max_train=None` and `max_test=None`.

## Reproducibility rules

- Fixed random seed for shuffling and model fitting.
- Vectorizer fitted only on the training split.
- Accuracy, macro F1 and weighted F1 recorded together.
- Dataset source and license metadata stored with every result.
- Results written as machine-readable JSON under `reports/`.

Always review the current dataset card and original publisher terms before redistribution or commercial use.