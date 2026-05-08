"""Shared utilities for the Lasso Peptide Classifier pipeline.

Components:
  - LassoDataset: PyTorch Dataset loading pre-computed .pt tensor files.
  - EarlyStopping: configurable patience-based early stopping tracker.
  - compute_metrics: accuracy, precision, recall, F1, AUC calculation.
  - print_metrics: formatted one-line metric display.
  - evaluate_model: full evaluation loop (loss + metrics) on a DataLoader.
  - extract_esm2_embeddings: frozen ESM-2 inference on FASTA files.
  - load_esm_model: ESM-2 tokenizer + model loader from HuggingFace.
  - load_classifier_from_checkpoint: checkpoint loading with arch metadata support.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from config import MAX_LEN


class LassoDataset(Dataset):
    """PyTorch Dataset backed by a .pt file containing pre-computed ESM-2 embeddings.

    The .pt file is expected to have keys:
      - "X": float tensor (N, seq_len, embed_dim)
      - "y": float tensor (N,)
    """

    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.X = data["X"]
        self.y = data["y"].unsqueeze(1)  # (N,) → (N, 1) for BCEWithLogitsLoss
        if self.X.dim() != 3:
            raise ValueError(
                f"Expected X to have 3 dimensions (N, seq_len, embed_dim), "
                f"got shape {tuple(self.X.shape)}"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Tracks best validation score and stops when no improvement for `patience` calls.

    Usage:
        es = EarlyStopping(patience=10)
        for epoch in range(epochs):
            val_f1 = validate()
            es(val_f1)
            if es.early_stop:
                break

    Returns True from __call__ when a new best score is recorded, False otherwise.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta           # minimum improvement to count as "better"
        self.counter = 0                      # epochs since last improvement
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """Report a new validation score. Returns True if it's a new best."""
        if np.isnan(val_score):
            return False
        if self.best_score is None:
            self.best_score = val_score
            return False
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics from tensors.

    Args:
        y_true, y_pred, y_prob: torch tensors of shape (N, 1) or (N,).

    Returns:
        dict with keys: accuracy, precision, recall, f1, auc_roc.
        AUC is NaN when only one class is present.
    """
    y_true_np = y_true.cpu().numpy().ravel()
    y_pred_np = y_pred.cpu().numpy().ravel()
    y_prob_np = y_prob.cpu().numpy().ravel()
    n_classes = len(np.unique(y_true_np))
    auc = roc_auc_score(y_true_np, y_prob_np) if n_classes > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "f1": f1_score(y_true_np, y_pred_np, zero_division=0),
        "auc_roc": auc,
    }


def print_metrics(split, metrics):
    """Print a one-line summary of metrics for a given data split."""
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics["auc_roc"]) else " N/A"
    print(
        f"  [{split}] Acc: {metrics['accuracy']:.4f}  "
        f"Prec: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}  "
        f"AUC: {auc_str}"
    )


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, return_predictions=False):
    """Run a full evaluation pass over a DataLoader.

    Args:
        model: LassoPeptideClassifier (or compatible).
        loader: DataLoader yielding (X, y) batches.
        criterion: loss function (e.g. BCEWithLogitsLoss).
        device: torch device.
        return_predictions: if True, also return (y_true, y_pred, y_prob) tensors.

    Returns:
        If return_predictions=False: dict with keys loss, accuracy, precision, recall, f1, auc_roc.
        If return_predictions=True:  (metrics_dict, (y_true, y_pred, y_prob)) tuple.
    """
    model.eval()
    total_loss = 0.0
    all_y, all_pred, all_prob = [], [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_y.append(y_batch.cpu())
        all_pred.append(preds.cpu())
        all_prob.append(probs.cpu())

    y_true = torch.cat(all_y)
    y_pred = torch.cat(all_pred)
    y_prob = torch.cat(all_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    if return_predictions:
        return metrics, (y_true, y_pred, y_prob)
    return metrics


def extract_esm2_embeddings(fasta_file, model_name, model, tokenizer, device, batch_size, max_len=MAX_LEN):
    """Extract frozen ESM-2 per-token embeddings for all sequences in a FASTA file.

    Sequences are tokenized with padding="max_length" and truncation to max_len.
    Padding positions are masked to zero by multiplying with the attention mask.

    Args:
        fasta_file: path to input FASTA.
        model_name: HuggingFace ESM-2 model ID (for logging only).
        model, tokenizer: loaded ESM-2 model and tokenizer.
        device: torch device for inference.
        batch_size: number of sequences per inference batch.
        max_len: max sequence length (padded/truncated). Default MAX_LEN=100.

    Returns:
        (ids, embeddings) where ids is list[str] of sequence IDs and
        embeddings is a tensor of shape (N, max_len, embed_dim) on CPU.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    sequences = [str(r.seq) for r in records]
    ids = [r.id for r in records]
    if not records:
        raise ValueError(f"No sequences found in FASTA file: {fasta_file}")

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state                     # (B, max_len, embed_dim)
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, max_len, 1)
            hidden = hidden * mask                                  # zero out padding
            all_embeds.append(hidden.cpu())

    return ids, torch.cat(all_embeds, dim=0)


def load_esm_model(model_name, device=None):
    """Load an ESM-2 tokenizer and model from HuggingFace.

    Downloads the model on first call (cached thereafter).
    The model is set to eval mode and moved to the given device.

    Args:
        model_name: HuggingFace ESM-2 model ID (e.g. "facebook/esm2_t12_35M_UR50D").
        device: torch device. Auto-detects CUDA vs CPU if None.

    Returns:
        (model, tokenizer, device) tuple.
    """
    from transformers import EsmTokenizer, EsmModel
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name).to(device).eval()
    except (OSError, EnvironmentError) as e:
        raise RuntimeError(
            f"Failed to load ESM model '{model_name}'. "
            f"First use requires network access. Original error: {e}"
        ) from e
    return model, tokenizer, device


def load_classifier_from_checkpoint(ckpt_path, embed_dim=None, device=None):
    """Load a LassoPeptideClassifier from a checkpoint file.

    Supports two checkpoint formats:
      1. Full format: {"state_dict": ..., "arch": {embed_dim, cnn_channels, ...}}
      2. Legacy format: raw state_dict (requires passing embed_dim explicitly).

    Args:
        ckpt_path: path to the .pt checkpoint file.
        embed_dim: required for legacy checkpoints lacking arch metadata.
        device: torch device. Auto-detects if None.

    Returns:
        LassoPeptideClassifier instance in eval mode on the target device.

    Raises:
        FileNotFoundError: if ckpt_path does not exist.
        ValueError: if checkpoint lacks arch info and embed_dim is not provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    from model import LassoPeptideClassifier
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    ARCH_KEYS = ["embed_dim", "cnn_channels", "cnn_kernels", "attention_heads", "mlp_hidden", "dropout"]
    if isinstance(state, dict) and "arch" in state:
        # Full format with architecture metadata
        arch = state["arch"]
        missing = [k for k in ARCH_KEYS if k not in arch]
        if missing:
            raise ValueError(
                f"Checkpoint arch is missing keys: {missing}. "
                f"The checkpoint may be corrupted or from an older version."
            )
        model = LassoPeptideClassifier(
            embed_dim=arch["embed_dim"],
            cnn_channels=arch["cnn_channels"],
            cnn_kernels=arch["cnn_kernels"],
            attention_heads=arch["attention_heads"],
            mlp_hidden=arch["mlp_hidden"],
            dropout=arch["dropout"],
        ).to(device)
        model.load_state_dict(state["state_dict"])
    elif embed_dim is not None:
        # Legacy format: reconstruct with default architecture + given embed_dim
        model = LassoPeptideClassifier(embed_dim=embed_dim).to(device)
        model.load_state_dict(state)
    else:
        raise ValueError(
            "Checkpoint lacks architecture info and embed_dim was not provided. "
            "Re-save the checkpoint with arch metadata or pass embed_dim."
        )
    model.eval()
    return model
