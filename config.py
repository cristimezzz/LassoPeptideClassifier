"""Central configuration for the Lasso Peptide Classifier project.

All paths, hyperparameters, ESM-2 model options, and helper functions are defined here.
Import from this module to use shared constants and utilities across all scripts.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────
DATA_DIR = "./data"                  # raw FASTA files
DATASET_DIR = "./dataset"            # pre-processed .pt tensor files
CHECKPOINT_DIR = "./checkpoints"     # model checkpoints
RESULTS_DIR = "./results"            # evaluation plots and reports

RAW_POS = os.path.join(DATA_DIR, "raw_positives.fasta")
RAW_NEG = os.path.join(DATA_DIR, "raw_negatives.fasta")
CLEAN_POS = os.path.join(DATA_DIR, "clean_positives.fasta")       # after CD-HIT dedup
CLEAN_NEG = os.path.join(DATA_DIR, "clean_negatives.fasta")       # after CD-HIT dedup

# ─── ESM-2 Model Choices ──────────────────────────────
# Each entry: name (HuggingFace model ID), dim (embedding dimension),
# batch (recommended ESM-2 inference batch size), vram (estimated GPU memory).
ESM_MODEL_CHOICES = [
    {
        "idx": 1,
        "name": "facebook/esm2_t6_8M_UR50D",
        "label": "ESM-2 8M",
        "params": "8M",
        "dim": 320,
        "vram": "≥ 2 GB",
        "batch": 8,
    },
    {
        "idx": 2,
        "name": "facebook/esm2_t12_35M_UR50D",
        "label": "ESM-2 35M",
        "params": "35M",
        "dim": 480,
        "vram": "≥ 4 GB",
        "batch": 4,
    },
    {
        "idx": 3,
        "name": "facebook/esm2_t30_150M_UR50D",
        "label": "ESM-2 150M",
        "params": "150M",
        "dim": 640,
        "vram": "≥ 8 GB",
        "batch": 2,
    },
    {
        "idx": 4,
        "name": "facebook/esm2_t33_650M_UR50D",
        "label": "ESM-2 650M",
        "params": "650M",
        "dim": 1280,
        "vram": "≥ 16 GB",
        "batch": 1,
    },
]

ESM_MODEL_NAME = ESM_MODEL_CHOICES[1]["name"]  # default: esm2_t12_35M (balance of speed/accuracy)

# ─── Data ─────────────────────────────────────────────
MAX_LEN = 100                # max sequence length: truncate longer, pad shorter
CD_HIT_THRESHOLD = 0.5       # sequence identity threshold for CD-HIT clustering
UNIPROT_NEG_LIMIT = 20000    # max negative sequences to fetch from UniProt
RANDOM_SEED = 42             # base random seed for reproducibility

# ─── Training ─────────────────────────────────────────
BATCH_SIZE = 32              # classifier training batch size
EPOCHS = 100                 # max training epochs (early stopping may stop earlier)
LR = 1e-4                    # AdamW learning rate
WEIGHT_DECAY = 1e-5          # AdamW weight decay (L2 regularization)
PATIENCE = 10                # early stopping patience (epochs without improvement)
DROPOUT = 0.3                # dropout rate in MLP classifier head

# ─── Model Architecture ───────────────────────────────
CNN_CHANNELS = [128, 128, 256]  # output channels for each Conv1D block
CNN_KERNELS = [5, 3, 3]         # kernel sizes (padding = k//2 preserves length before pooling)
ATTENTION_HEADS = 4             # number of heads in Multi-Head Attention
MLP_HIDDEN = 64                 # hidden dim in the MLP classifier head

# ─── Evaluation / Prediction ──────────────────────────
PRED_BATCH_SIZE = 64         # batch size for inference (can be larger than training)

# ─── Grid Search Space (used by run_experiment.py --strategy grid) ──
GRID_LR = [1e-3, 1e-4, 5e-5]
GRID_DROPOUT = [0.2, 0.3, 0.5]
GRID_CNN_CHANNELS = [[64, 128, 256], [128, 128, 256], [128, 256, 512]]
GRID_BATCH_SIZE = [16, 32]

# ─── Helpers ──────────────────────────────────────────

def get_esm_embed_dim(model_name=None):
    """Return the embedding dimension for a given ESM-2 model.

    Queries HuggingFace EsmConfig; will download config on first call if not cached.
    """
    from transformers import EsmConfig
    name = model_name or ESM_MODEL_NAME
    return EsmConfig.from_pretrained(name).hidden_size


def get_model_info(model_name=None):
    """Look up the ESM_MODEL_CHOICES entry for a model name.

    Returns the default model (index 1, t12_35M) if name is not found.
    """
    name = model_name or ESM_MODEL_NAME
    for m in ESM_MODEL_CHOICES:
        if m["name"] == name:
            return m
    return ESM_MODEL_CHOICES[1]


def ensure_dirs():
    """Create all required output directories if they don't exist."""
    for d in [DATA_DIR, DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def print_model_table():
    """Display a formatted table of available ESM-2 models with GPU info."""
    print("\n" + "=" * 70)
    print("  ESM-2 模型选择")
    print("-" * 70)
    print(f"  {'#':>3}  {'模型':<26} {'参数量':<8} {'dim':<6} {'推荐显存':<10} {'建议 batch':<10}")
    print("-" * 70)
    for m in ESM_MODEL_CHOICES:
        print(f"  {m['idx']:>3}  {m['label']:<26} {m['params']:<8} {m['dim']:<6} "
              f"{m['vram']:<10} {m['batch']:<10}")
    print("-" * 70)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    print(f"  当前 GPU 显存: {vram:.1f} GB" if vram else "  当前: CPU 模式")
    print("=" * 70 + "\n")


def select_esm_model(interactive=True):
    """Interactive ESM-2 model selection via CLI.

    Args:
        interactive: if False, return the default model immediately.

    Returns:
        (model_name, model_info_dict) tuple.
    """
    print_model_table()
    if not interactive:
        return ESM_MODEL_NAME, ESM_MODEL_CHOICES[1]

    while True:
        try:
            choice = input(f"  请选择模型编号 [1-{len(ESM_MODEL_CHOICES)}, 默认 2]: ").strip()
            if choice == "":
                choice = "2"
            idx = int(choice)
            for m in ESM_MODEL_CHOICES:
                if m["idx"] == idx:
                    return m["name"], m
            print("  [!] 无效编号，请重试\n")
        except ValueError:
            print("\n  [i] 使用默认模型\n")
            return ESM_MODEL_NAME, ESM_MODEL_CHOICES[1]
        except KeyboardInterrupt:
            print()
            raise
