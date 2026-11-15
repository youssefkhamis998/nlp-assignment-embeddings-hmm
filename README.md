# NLP Assignment 2: Embeddings and Hidden Markov Models

**Alexandria University**  
**Faculty of Engineering – Computer and Systems Department**  
**Course:** Natural Language Processing

## Overview

This project implements fundamental Natural Language Processing techniques from scratch using Python and NumPy.

The assignment explores sparse and dense word representations and sequence modeling using Hidden Markov Models (HMM).

## Assignment Parts

### Part 1: Sparse and Dense Word Representations

- Text preprocessing and vocabulary construction.
- Pointwise Mutual Information (PMI).
- TF-IDF representations.
- Skip-Gram Word2Vec with Negative Sampling.
- Word similarity and representation comparison.

### Part 2: Hidden Markov Model and Viterbi

- Initial, transition, and emission probabilities.
- Probability smoothing.
- Unknown word handling.
- Viterbi decoding.
- POS tagging evaluation.

### Part 3: Embedding-Assisted HMM

- PMI-assisted HMM.
- TF-IDF-assisted HMM.
- Word2Vec-assisted HMM.
- Comparison with baseline HMM.
- Tagging error analysis.

## Dataset

CoNLL-2000 dataset:

https://huggingface.co/datasets/eriktks/conll2000

The assignment uses the `tokens` and `pos_tags` fields.

## Technologies

- Python
- NumPy
- Matplotlib
- Hugging Face Datasets
- Jupyter Notebook

## Implementation Constraints

All core algorithms are implemented from scratch.

- No pretrained embeddings.
- No ready-made PMI or TF-IDF vectorizers.
- No existing Word2Vec implementations.
- No automatic differentiation or framework optimizers.
- No ready-made HMMs, Viterbi decoders, or POS taggers.

## Project Structure

- `notebooks/`: Main assignment notebook.
- `src/`: Supporting algorithm implementations.
- `data/`: Dataset storage.
- `outputs/`: Generated results and plots.

## Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```
