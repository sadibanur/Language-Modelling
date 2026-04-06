# NLP 243 Homework 2: Decoder-Only Language Model

Implementation of a GPT-style decoder-only Transformer for next-token prediction on the Penn Treebank dataset.

## Setup
```bash
pip install -r requirements.txt
```

## Project Structure

├── data/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── data.py              # GPT2 tokenization and DataLoaders
├── model.py             # Transformer model definition
├── train.py             # Training loop
├── evaluation.py        # Perplexity evaluation
├── main.py              # Entry point
├── util.py              # Helper functions
└── download_best_model.py  # Download best model from Google Drive

## Model Architecture

A 6-layer decoder-only Transformer with the following configuration:

| Hyperparameter | Value |
|---|---|
| Model dimension (`d_model`) | 384 |
| Attention heads (`n_heads`) | 6 |
| Feed-forward dimension (`d_ff`) | 1536 |
| Layers (`n_layers`) | 6 |
| Max sequence length | 256 |
| Tokenizer | GPT-2 (HuggingFace) |

Components:
- **Token + Sinusoidal Position Embeddings**
- **Multi-Head Causal Self-Attention** with padding mask
- **MLP** with GELU activation and residual connections
- **Pre-norm LayerNorm** at each block
- **Weight initialization** with normal distribution (mean=0, std=0.02)

## Training
```bash
python main.py
```

Trains for 8 epochs using AdamW (lr=3e-4) with cross-entropy loss. Padding tokens are ignored during loss computation. Best model checkpoint is saved based on validation loss.

## Evaluation

Download the best model first:
```bash
python download_best_model.py
```

Then evaluate:
```bash
python evaluation.py --model_path best_model.pt --batch_size 64
```

## Results

| Model | Train PPL | Val PPL | Test PPL |
|---|---|---|---|
| Model 1 (4 epochs, lr=3e-4) | 5.23 | 3.92 | 3.80 |
| **Model 2 (8 epochs, lr=3e-4)** | **2.85** | **3.70** | **3.61** |
| Model 3 (8 epochs, lr=5e-4) | 2.91 | 3.75 | 3.65 |

Model 2 is the best model with weight initialization and 8 training epochs.
