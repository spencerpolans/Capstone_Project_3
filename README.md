# Capstone Project 3: Dialogue Summarization with BART

## Project Overview

This project develops a text summarization model using a pre-trained Transformer architecture (BART) to generate concise summaries of messenger-like conversations. The model is trained and evaluated on the SAMSum dataset (test.csv, validation.csv, train.csv), which contains thousands of text-like dialogues paired with human summaries.

## Objective

The goal is to help users quickly digest lengthy or busy chat conversations by generating short, accurate summaries. This can reduce information overload, improve engagement, and support asynchronous communication for messaging apps.

## Tools & Libraries
Transformers
Datasets
PyTorch
Google Colab (to speed up training/GPU access if local computer has CPU limits)
evaluate (for ROUGE scoring)
Pandas
Tdqm

NOTE: tqdm is optional. Added in to see training progress.

## Methodology

### 1. Data Preparation
1) Loaded csvs from SAMSum.
2) Tokenized using `facebook/bart-base` tokenizer.
3) Padded and masked sequences appropriately.
4) Converted to `torch`-ready datasets.

### 2. Model Architecture
Fine-tuned the `facebook/bart-base` model (encoder-decoder architecture).
Experimented with freezing encoder for faster training (later unfroze for full performance).

### 3. Training
Trained on subsets (500, 2000) and full dataset with GPU acceleration.
Adjusted hyperparameters like `num_beams`, `max_length`, and `length_penalty` during generation.

### 4. Evaluation
Used ROUGE metrics to evaluate summary quality:
**ROUGE-1**: 50.49
**ROUGE-2**: 26.32
**ROUGE-L**: 41.78
**ROUGE-Lsum**: 41.83


## Key Learnings

Switching from BERT2BERT to BART significantly improved results.
GPU acceleration (Google Colab) was essential for training scalability.
Freezing the encoder helped with speed but limited final performance.
Padding must be carefully handled to avoid skewing loss.
Summary quality improves significantly with proper tuning and batching strategies.

## Business Impact

**Reduces cognitive load** for users catching up on conversations.
Enables **on-demand summarization** (e.g., daily or weekly digests).
Opens up **monetization opportunities** for productivity-focused platforms.

## Files Included

`Capstone_Project_3_manual_BART.ipynb`: Full training and evaluation notebook.
`train.csv`, `validation.csv`, `test.csv`: SAMSum datset.
`bart-samsum-2000-v1`: Saved model folder (not included in GitHub due to size). **To reproduce**: run the training loop in the .ipynb file on 5 epochs with the entire dataset.
