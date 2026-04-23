import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


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
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()
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
