# NLP Problem Catalog

Use this guide to choose a sensible first approach and a modern alternative.

| Problem | Recommended baseline | Modern extension | Primary metrics |
|---|---|---|---|
| Sentiment or intent classification | word/character TF-IDF + logistic regression | fine-tuned encoder or zero-shot model | macro F1, per-class recall |
| Spam and abuse detection | TF-IDF + linear SVM/logistic regression | transformer classifier | PR-AUC, recall at precision |
| Topic discovery | LDA with coherence review | BERTopic or embedding clustering | coherence plus human review |
| Semantic search | TF-IDF cosine similarity | sentence embeddings + vector index | Recall@K, MRR, nDCG |
| Named entity recognition | rules and dictionaries | token-classification transformer | entity-level precision/recall/F1 |
| Keyword extraction | TF-IDF or RAKE | embedding-based ranking | human relevance, precision@K |
| Text summarization | extractive sentence ranking | encoder-decoder or instruction model | ROUGE plus factuality review |
| Question answering | BM25 retrieval | retrieval-augmented generation | exact match, faithfulness, retrieval recall |
| Language detection | character n-grams | multilingual encoder | macro F1 |
| Duplicate detection | token similarity | sentence embeddings + threshold | ROC-AUC, F1 at chosen threshold |

## Model selection checklist

1. Define the business error costs before selecting a metric.
2. Build a deterministic baseline before using a large model.
3. Split data by time, user, or source when random splitting would leak information.
4. Measure latency, memory, throughput, and cost alongside accuracy.
5. Inspect errors across language, text length, class, and relevant user groups.
6. Calibrate thresholds using validation data rather than the test set.
7. Record model version, dataset version, parameters, and random seed.
8. Add human review and abstention for high-impact or low-confidence cases.

## Traditional versus modern is not either-or

A practical production system often combines both. Rules can protect known critical cases, a classical model can handle easy high-volume traffic, and a transformer can process ambiguous examples. This cascade can improve explainability, latency, and cost while retaining modern-model quality where it matters.
