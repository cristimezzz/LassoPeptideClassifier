"""Single-run training script for the Lasso Peptide Classifier.

Trains LassoPeptideClassifier on pre-computed ESM-2 embeddings from dataset/*.pt.
Uses BCEWithLogitsLoss with class-balanced pos_weight, AdamW optimizer, and
cosine annealing LR schedule. Saves the best checkpoint (by validation F1)
with architecture metadata for easy reloading.

Can run standalone (python train.py) or be imported by run_experiment.py
for multi-seed / CV / grid search strategies.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    DROPOUT,
    PATIENCE,
    CNN_CHANNELS,
    CNN_KERNELS,
    ATTENTION_HEADS,
    MLP_HIDDEN,
    ESM_MODEL_NAME,
    RANDOM_SEED,
    get_esm_embed_dim,
    ensure_dirs,
)
from model import LassoPeptideClassifier
from utils import LassoDataset, EarlyStopping, compute_metrics, print_metrics, evaluate_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch.

    Args:
        model: LassoPeptideClassifier.
        loader: DataLoader for training set.
        criterion: loss function (BCEWithLogitsLoss).
        optimizer: PyTorch optimizer.
        device: torch device.

    Returns:
        dict with keys loss, accuracy, precision, recall, f1, auc_roc for the epoch.
    """
    model.train()
    total_loss = 0.0
    all_y, all_pred, all_prob = [], [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        probs = torch.sigmoid(logits.detach())       # detach: metrics don't need grad
        preds = (probs >= 0.5).float()
        all_y.append(y_batch.cpu())
        all_pred.append(preds.cpu())
        all_prob.append(probs.cpu())

    y_true = torch.cat(all_y)
    y_pred = torch.cat(all_pred)
    y_prob = torch.cat(all_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, verbose=True, checkpoint_path=None, arch=None):
    """Full training loop with early stopping and checkpoint management.

    Tracks best model by validation F1. If checkpoint_path is provided, saves the
    best model to disk (with optional arch metadata). Otherwise keeps best state
    in memory via deepcopy.

    Args:
        model: LassoPeptideClassifier instance on target device.
        train_loader, val_loader: DataLoaders.
        criterion, optimizer, scheduler: training components.
        device: torch device.
        epochs: max epochs (early stopping may stop earlier).
        patience: early stopping patience (epochs without improvement).
        verbose: if True, print per-epoch metrics.
        checkpoint_path: path to save best model. If None, keeps in memory.
        arch: optional dict of architecture params to embed in checkpoint.

    Returns:
        (best_val_f1, best_epoch) tuple. best_val_f1 is -1.0 if no improvement.
    """
    early_stopping = EarlyStopping(patience=patience)
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None  # fallback for in-memory best model

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  ", end="")
            print_metrics("Train", train_metrics)
            print(f"             ", end="")
            print_metrics("Val", val_metrics)

        val_f1 = val_metrics["f1"]
        if np.isnan(val_f1):
            if verbose:
                print(f"     [!] Warning: NaN F1 at epoch {epoch}. Replacing with -1.0")
            val_f1 = -1.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            if checkpoint_path is not None:
                save_dict = {"state_dict": model.state_dict()}
                if arch is not None:
                    save_dict["arch"] = arch
                torch.save(save_dict, checkpoint_path)
            else:
                from copy import deepcopy
                best_state = deepcopy(model.state_dict())

        early_stopping(val_f1)
        if early_stopping.early_stop:
            if verbose:
                print(f"[*] Early stopping at epoch {epoch}")
            break

    # Restore best model weights
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    else:
        if verbose:
            print("[!] Warning: No best model found (all val scores may be NaN). "
                  "Keeping final parameters.")

    return best_val_f1, best_epoch


def main(esm_model=None, seed=None, checkpoint_name="best_model.pt", verbose=True,
         cnn_channels=None, cnn_kernels=None, dropout=None, lr=None, batch_size=None,
         device=None):
    """Entry point for a single training run.

    Loads the pre-split dataset, constructs the model with optionally overridden
    hyperparameters, computes class-balanced loss weights, and runs the training loop.

    Args:
        esm_model: HuggingFace ESM-2 model ID. Default from config.
        seed: random seed for reproducibility. Default from config.
        checkpoint_name: filename for the saved checkpoint. Default "best_model.pt".
        verbose: if True, print detailed training progress.
        cnn_channels, cnn_kernels, dropout, lr, batch_size: override config defaults
            (used by grid search in run_experiment.py).

    Returns:
        dict of test set metrics (accuracy, precision, recall, f1, auc_roc, loss).
    """
    ensure_dirs()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if esm_model is None:
        esm_model = ESM_MODEL_NAME
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        actual_seed = seed
    else:
        actual_seed = RANDOM_SEED
        torch.manual_seed(actual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    embed_dim = get_esm_embed_dim(esm_model)
    if verbose:
        print(f"[*] Device: {device}, embed_dim: {embed_dim}, seed: {actual_seed}")

    # Load pre-computed datasets
    for split in ("train", "val", "test"):
        path = os.path.join(DATASET_DIR, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Run data_pipeline.py first."
            )
    train_set = LassoDataset(os.path.join(DATASET_DIR, "train.pt"))
    val_set = LassoDataset(os.path.join(DATASET_DIR, "val.pt"))
    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))

    bs = batch_size or BATCH_SIZE
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    # Optional hyperparameter overrides (for grid search)
    _cnn_ch = cnn_channels or CNN_CHANNELS
    _cnn_ks = cnn_kernels or CNN_KERNELS
    _dropout = dropout if dropout is not None else DROPOUT
    _lr = lr if lr is not None else LR

    model = LassoPeptideClassifier(
        embed_dim=embed_dim,
        cnn_channels=_cnn_ch,
        cnn_kernels=_cnn_ks,
        attention_heads=ATTENTION_HEADS,
        mlp_hidden=MLP_HIDDEN,
        dropout=_dropout,
    ).to(device)

    # Class-balanced loss: pos_weight = neg_count / pos_count
    pos_count = (train_set.y == 1).sum().item()
    neg_count = (train_set.y == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Architecture metadata saved alongside state_dict for checkpoint portability
    arch = {
        "embed_dim": embed_dim,
        "cnn_channels": _cnn_ch,
        "cnn_kernels": _cnn_ks,
        "attention_heads": ATTENTION_HEADS,
        "mlp_hidden": MLP_HIDDEN,
        "dropout": _dropout,
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if verbose:
        print(f"[*] Training:   {len(train_set)} (pos:{int(pos_count)} neg:{int(neg_count)})")
        print(f"[*] Validation: {len(val_set)}")
        print(f"[*] Test:       {len(test_set)}")
        print(f"[*] Params:     {sum(p.numel() for p in model.parameters()):,}")
        print()

    best_val_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, EPOCHS, PATIENCE, verbose=verbose, checkpoint_path=ckpt_path, arch=arch,
    )

    if verbose:
        print(f"\n[*] Best: epoch {best_epoch}  (Val F1: {best_val_f1:.4f})")

    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    if verbose:
        print()
        print_metrics("Test", test_metrics)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-run training for Lasso Peptide Classifier")
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    main(esm_model=args.esm_model, seed=args.seed)
