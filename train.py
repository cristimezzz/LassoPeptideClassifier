import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    ESM_EMBED_DIM,
    CNN_CHANNELS,
    CNN_KERNELS,
    ATTENTION_HEADS,
    MLP_HIDDEN,
)
from model import LassoPeptideClassifier
from utils import EarlyStopping, compute_metrics, print_metrics


class LassoDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.X = data["X"]
        self.y = data["y"].unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    train_set = LassoDataset(os.path.join(DATASET_DIR, "train.pt"))
    val_set = LassoDataset(os.path.join(DATASET_DIR, "val.pt"))
    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = LassoPeptideClassifier(
        embed_dim=ESM_EMBED_DIM,
        cnn_channels=CNN_CHANNELS,
        cnn_kernels=CNN_KERNELS,
        attention_heads=ATTENTION_HEADS,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    pos_count = (train_set.y == 1).sum().item()
    neg_count = (train_set.y == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_val_f1 = -1.0
    best_epoch = 0

    print(f"[*] Training samples: {len(train_set)} (pos:{int(pos_count)} neg:{int(neg_count)})")
    print(f"[*] Validation samples: {len(val_set)}")
    print(f"[*] Test samples: {len(test_set)}")
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS}  ", end="")
        print_metrics("Train", train_metrics)
        print(f"            ", end="")
        print_metrics("Val", val_metrics)

        val_f1 = val_metrics["f1"]
        if np.isnan(val_f1):
            val_f1 = -1.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"[*] Early stopping triggered at epoch {epoch}")
            break

    print(f"\n[*] Best model at epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_path):
        print("[-] No best model saved, using current model for test evaluation")
        best_path = None

    if best_path:
        model.load_state_dict(torch.load(best_path, map_location=device))
    print()
    test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics("Test", test_metrics)


if __name__ == "__main__":
    main()
