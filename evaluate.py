import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

from config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    ESM_EMBED_DIM,
    CNN_CHANNELS,
    CNN_KERNELS,
    ATTENTION_HEADS,
    MLP_HIDDEN,
    DROPOUT,
)
from model import LassoPeptideClassifier
from train import LassoDataset, evaluate
from utils import print_metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[+] Saved: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[+] Saved: {save_path}")


def plot_pr_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[+] Saved: {save_path}")


def plot_probability_distribution(y_true, y_prob, save_path):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    plt.figure(figsize=(8, 4))
    plt.hist(
        y_prob[y_true == 0],
        bins=40,
        alpha=0.6,
        label="Negative",
        color="steelblue",
        edgecolor="white",
    )
    plt.hist(
        y_prob[y_true == 1],
        bins=40,
        alpha=0.6,
        label="Positive",
        color="firebrick",
        edgecolor="white",
    )
    plt.axvline(0.5, color="gray", linestyle="--", label="Threshold=0.5")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[+] Saved: {save_path}")


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).float()
        all_y.append(y_batch)
        all_pred.append(preds)
        all_prob.append(probs)
    return torch.cat(all_y), torch.cat(all_pred), torch.cat(all_prob)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = LassoPeptideClassifier(
        embed_dim=ESM_EMBED_DIM,
        cnn_channels=CNN_CHANNELS,
        cnn_kernels=CNN_KERNELS,
        attention_heads=ATTENTION_HEADS,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[-] No checkpoint found at {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[*] Loaded checkpoint: {ckpt_path}")

    criterion = torch.nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    print()
    print_metrics("Test", test_metrics)

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
    y_true_np = y_true.numpy().ravel()
    y_pred_np = y_pred.numpy().ravel()
    y_prob_np = y_prob.numpy().ravel()

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(
        classification_report(
            y_true_np,
            y_pred_np,
            target_names=["Negative", "Positive"],
            digits=4,
        )
    )

    report = classification_report(
        y_true_np, y_pred_np, target_names=["Negative", "Positive"], digits=4, output_dict=True
    )
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(
            classification_report(
                y_true_np, y_pred_np, target_names=["Negative", "Positive"], digits=4
            )
        )
    print(f"[+] Saved: {report_path}")

    plot_confusion_matrix(y_true_np, y_pred_np, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_true_np, y_prob_np, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_pr_curve(y_true_np, y_prob_np, os.path.join(RESULTS_DIR, "pr_curve.png"))
    plot_probability_distribution(
        y_true_np, y_prob_np, os.path.join(RESULTS_DIR, "probability_distribution.png")
    )

    print(f"\n[+] All evaluation results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
