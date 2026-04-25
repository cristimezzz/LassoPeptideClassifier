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
        probs = torch.sigmoid(logits.detach())
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


def main(esm_model=None, seed=None, checkpoint_name="best_model.pt", verbose=True,
         cnn_channels=None, cnn_kernels=None, dropout=None, lr=None, batch_size=None):
    ensure_dirs()
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

    embed_dim = get_esm_embed_dim(esm_model)
    if verbose:
        print(f"[*] Device: {device}, embed_dim: {embed_dim}, seed: {actual_seed}")

    train_set = LassoDataset(os.path.join(DATASET_DIR, "train.pt"))
    val_set = LassoDataset(os.path.join(DATASET_DIR, "val.pt"))
    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))

    bs = batch_size or BATCH_SIZE
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

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

    pos_count = (train_set.y == 1).sum().item()
    neg_count = (train_set.y == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_val_f1 = -1.0
    best_epoch = 0

    if verbose:
        print(f"[*] Training:   {len(train_set)} (pos:{int(pos_count)} neg:{int(neg_count)})")
        print(f"[*] Validation: {len(val_set)}")
        print(f"[*] Test:       {len(test_set)}")
        print(f"[*] Params:     {sum(p.numel() for p in model.parameters()):,}")
        print()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        if verbose:
            print(f"Epoch {epoch:3d}/{EPOCHS}  ", end="")
            print_metrics("Train", train_metrics)
            print(f"             ", end="")
            print_metrics("Val", val_metrics)

        val_f1 = val_metrics["f1"]
        if np.isnan(val_f1):
            val_f1 = -1.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, checkpoint_name))

        early_stopping(val_f1)
        if early_stopping.early_stop:
            if verbose:
                print(f"[*] Early stopping at epoch {epoch}")
            break

    if verbose:
        print(f"\n[*] Best: epoch {best_epoch}  (Val F1: {best_val_f1:.4f})")

    best_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    elif verbose:
        print("[-] No best model saved, using current model for test evaluation")

    test_metrics = evaluate_model(model, test_loader, criterion, device)
    if verbose:
        print()
        print_metrics("Test", test_metrics)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    main(esm_model=args.esm_model, seed=args.seed)
