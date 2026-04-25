import os
import torch

# ─── Paths ────────────────────────────────────────────
DATA_DIR = "./data"
DATASET_DIR = "./dataset"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

RAW_POS = os.path.join(DATA_DIR, "raw_positives.fasta")
RAW_NEG = os.path.join(DATA_DIR, "raw_negatives.fasta")
CLEAN_POS = os.path.join(DATA_DIR, "clean_positives.fasta")
CLEAN_NEG = os.path.join(DATA_DIR, "clean_negatives.fasta")

# ─── ESM-2 Model Choices ──────────────────────────────
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

ESM_MODEL_NAME = ESM_MODEL_CHOICES[1]["name"]  # default: t12_35M
ESM_BATCH_SIZE = ESM_MODEL_CHOICES[1]["batch"]

# ─── Data ─────────────────────────────────────────────
MAX_LEN = 100
CD_HIT_THRESHOLD = 0.5
UNIPROT_NEG_LIMIT = 20000
RANDOM_SEED = 42

# ─── Training ─────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10
DROPOUT = 0.3

# ─── Model Architecture ───────────────────────────────
CNN_CHANNELS = [128, 128, 256]
CNN_KERNELS = [5, 3, 3]
ATTENTION_HEADS = 4
MLP_HIDDEN = 64

# ─── Evaluation / Prediction ──────────────────────────
PRED_BATCH_SIZE = 64

# ─── Grid Search Space ────────────────────────────────
GRID_LR = [1e-3, 1e-4, 5e-5]
GRID_DROPOUT = [0.2, 0.3, 0.5]
GRID_CNN_CHANNELS = [[64, 128, 256], [128, 128, 256], [128, 256, 512]]
GRID_BATCH_SIZE = [16, 32]

# ─── Helpers ──────────────────────────────────────────

def get_esm_embed_dim(model_name=None):
    from transformers import EsmConfig
    name = model_name or ESM_MODEL_NAME
    return EsmConfig.from_pretrained(name).hidden_size


def get_model_info(model_name=None):
    name = model_name or ESM_MODEL_NAME
    for m in ESM_MODEL_CHOICES:
        if m["name"] == name:
            return m
    return ESM_MODEL_CHOICES[1]


def ensure_dirs():
    for d in [DATA_DIR, DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def print_model_table():
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
        except (ValueError, KeyboardInterrupt):
            print("\n  [i] 使用默认模型\n")
            return ESM_MODEL_NAME, ESM_MODEL_CHOICES[1]
