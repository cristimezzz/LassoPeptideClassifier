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
    BATCH_SIZE,
    ESM_MODEL_NAME,
    CNN_CHANNELS,
    CNN_KERNELS,
    ATTENTION_HEADS,
    MLP_HIDDEN,
    DROPOUT,
    get_esm_embed_dim,
    ensure_dirs,
)
from utils import LassoDataset, evaluate_model, print_metrics, load_classifier_from_checkpoint


def plot_confusion_matrix(y_true, y_pred, save_path):
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


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).float()
        all_y.append(y_batch); all_pred.append(preds); all_prob.append(probs)
    return (torch.cat(all_y).numpy().ravel(),
            torch.cat(all_pred).numpy().ravel(),
            torch.cat(all_prob).numpy().ravel())


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = get_esm_embed_dim(ESM_MODEL_NAME)
    print(f"[*] Device: {device}")

    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model = load_classifier_from_checkpoint(ckpt_path, embed_dim, device)
    print(f"[*] Loaded checkpoint: {ckpt_path}")

    criterion = torch.nn.BCEWithLogitsLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print()
    print_metrics("Test", test_metrics)

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    report_text = classification_report(
        y_true, y_pred, target_names=["Negative", "Positive"], digits=4
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report_text)

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


if __name__ == "__main__":
    main()
