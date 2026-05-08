"""Test-set evaluation and visualization for a trained Lasso Peptide Classifier.

Loads a saved checkpoint, runs inference on test.pt, and produces:
  - Classification metrics (accuracy, precision, recall, F1, AUC) via evaluate_model.
  - sklearn classification_report (per-class precision/recall/F1).
  - Four visualization plots saved to results/:
      1. Confusion matrix (seaborn heatmap)
      2. ROC curve with AUC annotation
      3. Precision-Recall curve
      4. Predicted probability distribution histogram

Metrics and predictions are collected in a single forward pass via
evaluate_model(return_predictions=True) to avoid duplicate computation.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    PRED_BATCH_SIZE,
    ESM_MODEL_NAME,
    get_esm_embed_dim,
    get_model_info,
    ensure_dirs,
)
from utils import LassoDataset, evaluate_model, print_metrics, load_classifier_from_checkpoint


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true, y_pred: 1D numpy arrays of integer labels (0/1).
        save_path: path to save the PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Generate and save an ROC curve with AUC annotation.

    Args:
        y_true: 1D numpy array of true binary labels.
        y_prob: 1D numpy array of predicted probabilities for the positive class.
        save_path: path to save the PNG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path):
    """Generate and save a Precision-Recall curve.

    Args:
        y_true: 1D numpy array of true binary labels.
        y_prob: 1D numpy array of predicted probabilities.
        save_path: path to save the PNG.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_probability_distribution(y_true, y_prob, save_path):
    """Generate and save a histogram of predicted probabilities by true class.

    Args:
        y_true: 1D numpy array of true binary labels (0=negative, 1=positive).
        y_prob: 1D numpy array of predicted probabilities.
        save_path: path to save the PNG.
    """
    yt, yp = np.array(y_true).ravel(), np.array(y_prob).ravel()
    plt.figure(figsize=(8, 4))
    plt.hist(yp[yt == 0], bins=40, alpha=0.6, label="Negative", color="steelblue", edgecolor="white")
    plt.hist(yp[yt == 1], bins=40, alpha=0.6, label="Positive", color="firebrick", edgecolor="white")
    plt.axvline(0.5, color="gray", ls="--", label="Threshold=0.5")
    plt.xlabel("Predicted Probability"); plt.ylabel("Count")
    plt.title("Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(esm_model=None, checkpoint_name="best_model.pt", save_plots=True, device=None):
    """Load a checkpoint, evaluate on the test set, and generate plots.

    Args:
        esm_model: HuggingFace ESM-2 model ID. Default from config.
        checkpoint_name: checkpoint filename in checkpoints/. Default "best_model.pt".
        save_plots: if True, save PNGs and classification report to results/.

    Returns:
        dict of test metrics.
    """
    ensure_dirs()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if esm_model is None:
        esm_model = ESM_MODEL_NAME
    embed_dim = get_esm_embed_dim(esm_model)
    info = get_model_info(esm_model)
    print(f"[*] Device: {device}  |  ESM: {info['label']}  |  dim={embed_dim}")

    test_path = os.path.join(DATASET_DIR, "test.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}. Run data_pipeline.py first."
        )
    test_set = LassoDataset(test_path)
    test_loader = DataLoader(test_set, batch_size=PRED_BATCH_SIZE, shuffle=False)

    if len(test_set) == 0:
        print("[!] Test dataset is empty, skipping evaluation.")
        return {}

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    model = load_classifier_from_checkpoint(ckpt_path, embed_dim, device)
    print(f"[*] Loaded checkpoint: {ckpt_path}")

    # Single forward pass: compute metrics + collect raw predictions for plotting
    criterion = torch.nn.BCEWithLogitsLoss()
    test_metrics, (y_true_t, y_pred_t, y_prob_t) = evaluate_model(
        model, test_loader, criterion, device, return_predictions=True
    )
    y_true = y_true_t.numpy().ravel()
    y_pred = y_pred_t.numpy().ravel()
    y_prob = y_prob_t.numpy().ravel()
    print()
    print_metrics("Test", test_metrics)

    report_text = classification_report(
        y_true, y_pred, target_names=["Negative", "Positive"], digits=4
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report_text)

    if save_plots:
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(report_text)

        plot_confusion_matrix(y_true, y_pred, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plot_roc_curve(y_true, y_prob, os.path.join(RESULTS_DIR, "roc_curve.png"))
        plot_pr_curve(y_true, y_prob, os.path.join(RESULTS_DIR, "pr_curve.png"))
        plot_probability_distribution(y_true, y_prob, os.path.join(RESULTS_DIR, "probability_distribution.png"))
        print(f"\n[+] All results saved → {RESULTS_DIR}/")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Lasso Peptide Classifier on the test set")
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    parser.add_argument("--checkpoint", default="best_model.pt", help="Checkpoint filename in checkpoints/")
    args = parser.parse_args()
    main(esm_model=args.esm_model, checkpoint_name=args.checkpoint)
