import os

# ─── Paths ────────────────────────────────────────────
DATA_DIR = "./data"
DATASET_DIR = "./dataset"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

RAW_POS = os.path.join(DATA_DIR, "raw_positives.fasta")
RAW_NEG = os.path.join(DATA_DIR, "raw_negatives.fasta")
CLEAN_POS = os.path.join(DATA_DIR, "clean_positives.fasta")
CLEAN_NEG = os.path.join(DATA_DIR, "clean_negatives.fasta")

# ─── ESM-2 ────────────────────────────────────────────
# 可选模型: t6_8M (dim=320), t12_35M (dim=480), t30_150M (dim=640)
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM_BATCH_SIZE = 4  # 4GB 显存下保守值

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


def get_esm_embed_dim(model_name=None):
    """自动从模型配置中获取 ESM-2 的隐藏维度"""
    from transformers import EsmConfig
    name = model_name or ESM_MODEL_NAME
    return EsmConfig.from_pretrained(name).hidden_size


def ensure_dirs():
    for d in [DATA_DIR, DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
