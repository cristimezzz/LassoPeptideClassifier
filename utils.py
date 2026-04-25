import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from config import MAX_LEN


class LassoDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.X = data["X"]
        self.y = data["y"].unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if np.isnan(val_score):
            return False
        if self.best_score is None:
            self.best_score = val_score
            return True
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def compute_metrics(y_true, y_pred, y_prob):
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
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics["auc_roc"]) else " N/A"
    print(
        f"  [{split}] Acc: {metrics['accuracy']:.4f}  "
        f"Prec: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}  "
        f"AUC: {auc_str}"
    )


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
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


def extract_esm2_embeddings(fasta_file, model_name, model, tokenizer, device, batch_size, max_len=MAX_LEN):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    sequences = [str(r.seq) for r in records]
    ids = [r.id for r in records]

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
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = hidden * mask
            all_embeds.append(hidden.cpu())

    return ids, torch.cat(all_embeds, dim=0)


def load_esm_model(model_name, device=None):
    from transformers import EsmTokenizer, EsmModel
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer, device


def load_classifier_from_checkpoint(ckpt_path, embed_dim, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    from model import LassoPeptideClassifier
    model = LassoPeptideClassifier(embed_dim=embed_dim).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
