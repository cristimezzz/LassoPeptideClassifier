# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI classifier for lasso peptides in gastrointestinal proteins. Input: FASTA protein sequences. Output: probability [0,1] of being a lasso peptide. The project is bilingual (Chinese + English).

## Architecture

```
FASTA → ESM-2 (frozen) → 1D Conv ×3 → Multi-Head Attention → MLP → Probability
```

**Data pipeline**: LassoPred API (positive samples) + UniProt REST API (negative samples from bacteria, length 40-100, Swiss-Prot) → CD-HIT dedup (with pure-Python fallback) → ESM-2 frozen inference extracts embeddings → negatives downsampled to 3× positives → 80/10/10 stratified split → `.pt` tensors saved to `dataset/`.

**Model** (`model.py`): Three stacked Conv1D blocks (BatchNorm → ReLU → MaxPool1d) on transposed ESM-2 embeddings. Padding mask is max-pooled in parallel through the CNN stack. MultiheadAttention with residual connection + LayerNorm, then global mean pooling, then a 2-layer MLP (Dropout → Linear → ReLU → Dropout → Linear → 1). ~736K trainable parameters with t12_35M.

**Script pipeline**: Each stage is a standalone CLI script that can run independently, or be orchestrated by `run_experiment.py`:
1. `data_pipeline.py` — fetch from APIs + CD-HIT dedup + ESM-2 extraction + dataset splitting (also supports `--download-only`)
2. `train.py` — single-run training (BCEWithLogitsLoss with class-balanced pos_weight, AdamW, CosineAnnealingLR)
3. `evaluate.py` — test metrics + 4 plots (confusion matrix, ROC, PR, probability distribution)
4. `predict.py` — FASTA → CSV inference

**Jupyter notebook** (`lasso_peptide_classifier.ipynb`) is a self-contained interactive duplicate using a separate namespace (`lasso_data/`, etc.) to avoid collision.

## Common Commands

```bash
# One-click batch experiment (downloads data, extracts features, trains)
python run_experiment.py                                    # interactive ESM selection
python run_experiment.py --strategy multi_seed --runs 5     # multi-seed training
python run_experiment.py --strategy cv --folds 5            # K-fold cross-validation
python run_experiment.py --strategy grid                    # hyperparameter grid search
python run_experiment.py --skip-pipeline                    # skip data prep (use cached .pt)

# Step-by-step
python data_pipeline.py --download-only                         # fetch positive + negative samples only
python data_pipeline.py --esm-model facebook/esm2_t12_35M_UR50D  # CD-HIT + ESM extraction
python train.py --esm-model facebook/esm2_t12_35M_UR50D --seed 42
python evaluate.py --esm-model facebook/esm2_t12_35M_UR50D
python predict.py -i sequences.fasta -o results.csv

# Jupyter
jupyter notebook lasso_peptide_classifier.ipynb
```

## Key Configuration

All hyperparameters and paths live in `config.py`. Four ESM-2 models are available (8M/35M/150M/650M), defaulting to `esm2_t12_35M_UR50D` (embed_dim=480). Script-level overrides use `--esm-model` and `--seed` args.

Important config variables: `MAX_LEN=100`, `BATCH_SIZE=32`, `EPOCHS=100`, `LR=1e-4`, `DROPOUT=0.3`, `PATIENCE=10`, `CNN_CHANNELS=[128,128,256]`, `ATTENTION_HEADS=4`, `MLP_HIDDEN=64`. Grid search spaces are defined in `GRID_LR`, `GRID_DROPOUT`, `GRID_CNN_CHANNELS`, `GRID_BATCH_SIZE`.

## Checkpoint Format

Checkpoints saved by `train.py` include both `state_dict` and `arch` metadata (embed_dim, cnn_channels, cnn_kernels, attention_heads, mlp_hidden, dropout). `load_classifier_from_checkpoint` in `utils.py` handles both formats, with a fallback for legacy checkpoints lacking arch metadata (requires passing `embed_dim`).

## Platform Notes

- **Windows**: CD-HIT has no native binary; the pure-Python fallback in `data_pipeline.py` (greedy clustering with BioPython PairwiseAligner) runs automatically. PyTorch installation requires explicit CUDA version selection.
- **ESM-2 models**: Auto-downloaded from HuggingFace on first use. Frozen inference only — VRAM requirements are lower than training.
