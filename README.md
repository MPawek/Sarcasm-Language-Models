# Sarcasm Detection Models

This repository contains code for training and evaluating sarcasm detection models using:

- TF-IDF + Logistic Regression
- BERT (`bert-base-uncased`)

The project compares lexical and contextual models for sarcasm detection and evaluates how performance changes when surface-level cues such as punctuation and capitalization are removed.

## Project Overview

The main goal is to test whether sarcasm detection models rely on:

- lexical surface cues
- punctuation
- capitalization
- contextual meaning

Models are trained on Reddit sarcasm data and evaluated on both Reddit and Twitter data for cross-domain testing.

## Files

### `clean_data.py`

Preprocesses the Reddit dataset.

It creates the Reddit train, validation, and test files in several versions:

- clean text
- lowercased text
- punctuation-removed text
- lowercased + punctuation-removed text

Example output files:

- `train_processed_clean.csv`
- `train_processed_lowercase.csv`
- `train_processed_nopunct.csv`
- `train_processed_lower_nopunct.csv`
- `validation_processed_clean.csv`
- `test_processed_clean.csv`
- `test_processed_lowercase.csv`
- `test_processed_nopunct.csv`
- `test_processed_lower_nopunct.csv`

### `process_twitter.py`

Preprocesses the Twitter dataset for cross-domain evaluation.

It creates:

- `twitter_processed_clean.csv`
- `twitter_processed_lowercase.csv`
- `twitter_processed_nopunct.csv`
- `twitter_processed_lower_nopunct.csv`

### `TF_IDF.py`

Trains and evaluates the TF-IDF + Logistic Regression model.

Main features:

- TF-IDF vectorization
- unigram, bigram, and trigram features
- logistic regression classifier
- evaluation across ablated test sets
- optional evaluation-only mode for Twitter testing
- feature-weight inspection

### `BERT.py`

Fine-tunes and evaluates a BERT sarcasm classifier.

Main features:

- `bert-base-uncased`
- binary sarcasm classification
- validation-based checkpoint selection
- GPU support through PyTorch
- mixed precision support
- evaluation across ablated test sets
- optional evaluation-only mode for Twitter testing

### `graph.py`

Creates figures for comparing model performance.

Generated figures may include:

- F1 score across test conditions
- F1 drop after punctuation removal
- BERT precision vs. recall
- Reddit-to-Twitter cross-domain performance

## Data Format

The processed data should contain at least the following columns:

```text
processed_text,label
```

Where:

- `processed_text` is the cleaned text input
- `label` is the binary class label:
  - `1` = sarcastic
  - `0` = non-sarcastic

## Running the Pipeline

### 1. Preprocess Reddit data

```bash
python clean_data.py
```

### 2. Train and evaluate TF-IDF

```bash
python TF_IDF.py
```

### 3. Train and evaluate BERT

```bash
python BERT.py
```

### 4. Preprocess Twitter data

```bash
python process_twitter.py
```

### 5. Run cross-domain evaluation

Set evaluation-only mode in the model scripts, then run:

```bash
python TF_IDF.py
python BERT.py
```

### 6. Generate graphs

```bash
python graph.py
```

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- PR AUC

F1 score is used as the primary comparison metric because it balances precision and recall.

## Experimental Conditions

Each model is evaluated on four text variants:

| Condition | Description |
|---|---|
| Clean | Sarcasm tags removed, punctuation and capitalization preserved |
| Lowercase | Text converted to lowercase |
| No Punctuation | Punctuation removed |
| Lowercase + No Punctuation | Both transformations applied |

## Main Findings

The experiments found that:

- BERT outperforms TF-IDF on Reddit sarcasm detection.
- Capitalization has little to no measurable effect.
- Punctuation affects both models, especially BERT.
- BERT is more sensitive to punctuation removal than TF-IDF on Reddit.
- Cross-domain testing on Twitter shows performance degradation for both models.
- The effect of punctuation differs across domains, suggesting models learn dataset-specific cues.

## Requirements

Recommended Python packages:

```bash
pip install pandas numpy scikit-learn torch transformers matplotlib joblib tqdm
```

For GPU acceleration with NVIDIA CUDA, install the appropriate PyTorch version from:

```text
https://pytorch.org/get-started/locally/
```

## Notes

Large files such as datasets, trained model checkpoints, generated CSV outputs, and generated graph images may be excluded from the repository depending on storage constraints.

Recommended exclusions for `.gitignore` include:

```gitignore
*.csv
*.joblib
bert_model_output/
bert_best_model/
*.png
__pycache__/
.venv/
venv/
```

## Disclaimer

This README was generated with assistance from ChatGPT and reviewed/edited for use with this project.
