#!/usr/bin/env python3
# (Unix shebang; harmless on Windows — use `python run_experiment.py`)
"""One-click batch experiment runner for the Lasso Peptide Classifier.

Supports three training strategies:
  1. multi_seed — N runs with different random seeds, reports mean ± std.
  2. cv         — K-fold cross-validation on merged train+val, per-fold test metrics.
  3. grid       — hyperparameter grid search (LR × dropout × CNN channels × batch size).

Also orchestrates the full data pipeline (download → CD-HIT → ESM extraction → train)
when --skip-pipeline is not specified.

Usage:
  python run_experiment.py --strategy multi_seed --runs 5
  python run_experiment.py --strategy cv --folds 5
  python run_experiment.py --strategy grid
  python run_experiment.py --strategy multi_seed --esm-model facebook/esm2_t6_8M_UR50D --runs 3
"""

import argparse
import os
import shutil
import sys
import torch
import numpy as np
from datetime import datetime
from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from config import (
    ESM_MODEL_NAME,
    DATASET_DIR,
    CHECKPOINT_DIR,
    RANDOM_SEED,
    MAX_LEN,
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
    GRID_LR,
    GRID_DROPOUT,
    GRID_CNN_CHANNELS,
    GRID_BATCH_SIZE,
    print_model_table,
    select_esm_model,
    get_model_info,
    get_esm_embed_dim,
    ensure_dirs,
)
from model import LassoPeptideClassifier
from utils import (
    LassoDataset,
    compute_metrics,
    evaluate_model,
    load_esm_model,
    extract_esm2_embeddings,
)
import train as train_module


# ═══════════════════════════════════════════════════════════
#  Strategy 1: multi_seed — multi-random-seed training
# ═══════════════════════════════════════════════════════════

def run_multi_seed(esm_model, n_runs, base_seed=None):
    """Run N independent training runs with incremental random seeds.

    Each run uses seed = base_seed + i. The run with highest test F1 is copied
    to checkpoints/best_model.pt. Reports mean ± std of test F1 and AUC.

    Args:
        esm_model: HuggingFace ESM-2 model ID.
        n_runs: number of training runs.
        base_seed: starting random seed. Default from config.RANDOM_SEED.

    Returns:
        list of per-run result dicts.
    """
    if base_seed is None:
        base_seed = RANDOM_SEED

    info = get_model_info(esm_model)
    results = []

    print(f"\n{'=' * 60}")
    print(f"  Strategy: multi_seed  (n_runs={n_runs}, base_seed={base_seed})")
    print(f"  ESM:      {info['label']}  ({info['params']}, dim={info['dim']})")
    print(f"{'=' * 60}\n")

    best_f1 = -1.0
    best_seed = None
    all_f1, all_auc = [], []

    for i in range(n_runs):
        seed = base_seed + i
        print(f"[Run {i+1}/{n_runs}]  seed={seed}")
        print("-" * 40)

        metrics = train_module.main(
            esm_model=esm_model,
            seed=seed,
            checkpoint_name=f"run_{i}_model.pt",
            verbose=True,
        )

        f1 = metrics["f1"] if not np.isnan(metrics["f1"]) else 0.0
        auc = metrics["auc_roc"] if not np.isnan(metrics["auc_roc"]) else float("nan")
        all_f1.append(f1); all_auc.append(auc)
        results.append({"seed": seed, "f1": f1, "auc": auc, **metrics})

        if f1 > best_f1:
            best_f1 = f1; best_seed = seed
            src = os.path.join(CHECKPOINT_DIR, f"run_{i}_model.pt")
            dst = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            if os.path.exists(src):
                shutil.copy(src, dst)
        print()

    # Print summary statistics
    all_f1 = np.array(all_f1)
    all_auc = np.array([a for a in all_auc if not np.isnan(a)])
    print("=" * 60)
    print(f"  Summary ({n_runs} runs):")
    print(f"  Test F1:  {all_f1.mean():.4f} ± {all_f1.std():.4f}")
    if len(all_auc) > 0:
        print(f"  Test AUC: {all_auc.mean():.4f} ± {all_auc.std():.4f}")
    print(f"  Best:     seed={best_seed}  (F1={best_f1:.4f})")
    print(f"  Saved → {CHECKPOINT_DIR}/best_model.pt")
    print("=" * 60)
    return results


# ═══════════════════════════════════════════════════════════
#  Strategy 2: cv — K-fold cross-validation
# ═══════════════════════════════════════════════════════════

class CombinedDataset(Dataset):
    """Merges two .pt dataset files (train + val) into one Dataset for CV folds."""

    def __init__(self, pt_path_a, pt_path_b):
        da = torch.load(pt_path_a, map_location="cpu", weights_only=True)
        db = torch.load(pt_path_b, map_location="cpu", weights_only=True)
        self.X = torch.cat([da["X"], db["X"]], dim=0)
        self.y = torch.cat([da["y"].unsqueeze(1), db["y"].unsqueeze(1)], dim=0)

    def __len__(self):      return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def run_cv(esm_model, n_folds, seed=None):
    """K-fold cross-validation on merged train+val with hold-out test evaluation.

    For each fold, trains a fresh model on K-1 folds, validates on the held-out
    fold, and evaluates on the separate test set. Reports mean ± std across folds.

    Args:
        esm_model: HuggingFace ESM-2 model ID.
        n_folds: number of cross-validation folds.
        seed: random seed for fold splitting. Default from config.RANDOM_SEED.

    Returns:
        list of per-fold test metric dicts.
    """
    if seed is None:
        seed = RANDOM_SEED

    info = get_model_info(esm_model)
    embed_dim = get_esm_embed_dim(esm_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  Strategy: cv  (folds={n_folds})")
    print(f"  ESM:      {info['label']}  ({info['params']}, dim={info['dim']})")
    print(f"{'=' * 60}\n")

    combined = CombinedDataset(
        os.path.join(DATASET_DIR, "train.pt"),
        os.path.join(DATASET_DIR, "val.pt"),
    )
    test_set = LassoDataset(os.path.join(DATASET_DIR, "test.pt"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(combined)))):
        print(f"[Fold {fold+1}/{n_folds}]")
        print("-" * 40)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=val_sampler)

        model = LassoPeptideClassifier(
            embed_dim=embed_dim,
            cnn_channels=CNN_CHANNELS,
            cnn_kernels=CNN_KERNELS,
            attention_heads=ATTENTION_HEADS,
            mlp_hidden=MLP_HIDDEN,
            dropout=DROPOUT,
        ).to(device)

        # Class-balanced loss per fold (distribution may vary)
        pos_count = (combined.y[train_idx] == 1).sum().item()
        neg_count = len(train_idx) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_f1, _ = train_module.train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, EPOCHS, PATIENCE, verbose=True, checkpoint_path=None,
        )

        test_met = evaluate_model(model, test_loader, criterion, device)
        print(f"  Fold {fold+1} →  Val F1: {best_val_f1:.4f}  |  Test F1: {test_met['f1']:.4f}")
        fold_metrics.append(test_met)
        print()

    # Summary
    all_f1 = np.array([m["f1"] for m in fold_metrics])
    all_auc = np.array([m["auc_roc"] for m in fold_metrics if not np.isnan(m["auc_roc"])])
    print("=" * 60)
    print(f"  CV Summary ({n_folds} folds):")
    print(f"  Test F1:  {all_f1.mean():.4f} ± {all_f1.std():.4f}")
    if len(all_auc) > 0:
        print(f"  Test AUC: {all_auc.mean():.4f} ± {all_auc.std():.4f}")
    print("=" * 60)
    return fold_metrics


# ═══════════════════════════════════════════════════════════
#  Strategy 3: grid — hyperparameter grid search
# ═══════════════════════════════════════════════════════════

def run_grid_search(esm_model):
    """Exhaustive hyperparameter grid search over lr × dropout × cnn_channels × batch_size.

    Evaluates every combination in the Cartesian product of GRID_LR, GRID_DROPOUT,
    GRID_CNN_CHANNELS, and GRID_BATCH_SIZE (defined in config.py). Reports the
    Top-5 combinations ranked by test F1 and copies the best model to best_model.pt.

    Args:
        esm_model: HuggingFace ESM-2 model ID.

    Returns:
        list of result dicts sorted by F1 descending.
    """
    info = get_model_info(esm_model)

    combinations = list(product(GRID_LR, GRID_DROPOUT, GRID_CNN_CHANNELS, GRID_BATCH_SIZE))
    convert_cnn = lambda c: c if isinstance(c, list) else list(c)
    n_total = len(combinations)

    print(f"\n{'=' * 60}")
    print(f"  Strategy: grid search")
    print(f"  ESM:      {info['label']}  ({info['params']}, dim={info['dim']})")
    print(f"  Grid:     LR ∈ {GRID_LR}")
    print(f"            dropout ∈ {GRID_DROPOUT}")
    print(f"            CNN channels ∈ {GRID_CNN_CHANNELS}")
    print(f"            batch_size ∈ {GRID_BATCH_SIZE}")
    print(f"  Total:    {n_total} combinations")
    print(f"{'=' * 60}\n")

    results = []
    best_f1 = -1.0
    best_combo = None

    for i, (lr, dp, ch, bs) in enumerate(combinations):
        label = f"lr={lr:.0e} dp={dp} ch={ch} bs={bs}"
        print(f"[{i+1}/{n_total}] {label}")
        print("-" * 50)

        metrics = train_module.main(
            esm_model=esm_model,
            seed=RANDOM_SEED,
            checkpoint_name=f"grid_{i}_model.pt",
            verbose=False,                   # suppress per-epoch output for grid search
            lr=lr,
            dropout=dp,
            cnn_channels=convert_cnn(ch),
            batch_size=bs,
        )

        f1 = metrics["f1"] if not np.isnan(metrics["f1"]) else 0.0
        auc = metrics["auc_roc"] if not np.isnan(metrics["auc_roc"]) else float("nan")
        print(f"  Test F1: {f1:.4f}  AUC: {auc:.4f}\n")

        results.append({"lr": lr, "dropout": dp, "cnn_channels": ch, "batch_size": bs,
                         "f1": f1, "auc": auc})

        if f1 > best_f1:
            best_f1 = f1
            best_combo = results[-1]
            src = os.path.join(CHECKPOINT_DIR, f"grid_{i}_model.pt")
            dst = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            if os.path.exists(src):
                shutil.copy(src, dst)

    # Top-5 results
    results.sort(key=lambda x: x["f1"], reverse=True)
    print("=" * 60)
    print(f"  Top-5 Results:")
    print(f"  {'Rank':<5} {'F1':<10} {'AUC':<10} {'LR':<10} {'Dropout':<10} {'CNN_ch':<20} {'BS':<5}")
    print("  " + "-" * 55)
    for ri, r in enumerate(results[:5]):
        print(f"  {ri+1:<5} {r['f1']:<10.4f} {r['auc']:<10.4f} "
              f"{r['lr']:<10.0e} {r['dropout']:<10} {str(r['cnn_channels']):<20} {r['batch_size']:<5}")
    print(f"\n  Best combination:")
    print(f"  LR={best_combo['lr']}  dropout={best_combo['dropout']}  "
          f"channels={best_combo['cnn_channels']}  batch_size={best_combo['batch_size']}")
    print(f"  Saved → {CHECKPOINT_DIR}/best_model.pt")
    print("=" * 60)
    return results


# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(esm_model, force_redownload=False):
    """Execute the full data preparation pipeline: download + CD-HIT + ESM extraction.

    Called by main() before training unless --skip-pipeline is set.

    Args:
        esm_model: HuggingFace ESM-2 model ID.
        force_redownload: if True, re-download even if cache files exist.
    """
    print(f"[*] Preparing data for {esm_model} ...")
    print()

    # Step 1: download
    from data_pipeline import fetch_lassopred_to_fasta, fetch_uniprot_negatives
    from config import RAW_POS, RAW_NEG

    if force_redownload or not os.path.exists(RAW_POS):
        fetch_lassopred_to_fasta(RAW_POS)
    else:
        from Bio import SeqIO
        n = sum(1 for _ in SeqIO.parse(RAW_POS, "fasta"))
        print(f"[~] {RAW_POS} exists ({n} seqs)")

    if force_redownload or not os.path.exists(RAW_NEG):
        fetch_uniprot_negatives(RAW_NEG)
    else:
        from Bio import SeqIO
        n = sum(1 for _ in SeqIO.parse(RAW_NEG, "fasta"))
        print(f"[~] {RAW_NEG} exists ({n} seqs)")

    # Step 2: data pipeline (CD-HIT + ESM extraction)
    from data_pipeline import run_cd_hit, create_and_split_dataset
    from config import CLEAN_POS, CLEAN_NEG, CD_HIT_THRESHOLD

    run_cd_hit(RAW_POS, CLEAN_POS, threshold=CD_HIT_THRESHOLD)
    run_cd_hit(RAW_NEG, CLEAN_NEG, threshold=CD_HIT_THRESHOLD)
    create_and_split_dataset(CLEAN_POS, CLEAN_NEG, esm_model_name=esm_model)


def main():
    """CLI entry point: parse args, prepare data, run selected strategy."""
    parser = argparse.ArgumentParser(
        description="Lasso peptide classifier — batch experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --strategy multi_seed --runs 5
  python run_experiment.py --strategy cv --folds 5
  python run_experiment.py --strategy grid
  python run_experiment.py --strategy multi_seed --esm-model facebook/esm2_t6_8M_UR50D --runs 3
        """,
    )
    parser.add_argument("--strategy", default="multi_seed",
                        choices=["multi_seed", "cv", "grid"],
                        help="Training strategy (default: multi_seed)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs for multi_seed (default: 5)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds for cv (default: 5)")
    parser.add_argument("--esm-model", default=None,
                        help="ESM-2 model name (if not specified, interactive selection)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: config.RANDOM_SEED)")
    parser.add_argument("--redownload", action="store_true",
                        help="Force re-download data")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip data download & extraction (assumes .pt files exist)")
    args = parser.parse_args()

    ensure_dirs()

    # Model selection (explicit or interactive)
    if args.esm_model:
        esm_model = args.esm_model
        print_model_table()
        info = get_model_info(esm_model)
        print(f"[i] Using: {info['label']}  ({info['params']}, dim={info['dim']})")
    else:
        esm_model, info = select_esm_model(interactive=True)

    # Step 1-2: data preparation (once, shared across all strategies)
    if not args.skip_pipeline:
        run_full_pipeline(esm_model, force_redownload=args.redownload)
    else:
        if not os.path.exists(os.path.join(DATASET_DIR, "train.pt")):
            print("[-] --skip-pipeline specified but no .pt files found. Running pipeline anyway.")
            run_full_pipeline(esm_model, force_redownload=args.redownload)
        else:
            print("[~] Skipping pipeline, using existing dataset")

    # Step 3: strategy-specific training
    t0 = datetime.now()
    if args.strategy == "multi_seed":
        run_multi_seed(esm_model, args.runs, args.seed)
    elif args.strategy == "cv":
        run_cv(esm_model, args.folds, args.seed)
    elif args.strategy == "grid":
        run_grid_search(esm_model)

    print(f"\n[+] Total time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
