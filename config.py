import os

# ─── Paths ────────────────────────────────────────────
DATA_DIR = "./data"
DATASET_DIR = "./dataset"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_POS = os.path.join(DATA_DIR, "raw_positives.fasta")
RAW_NEG = os.path.join(DATA_DIR, "raw_negatives.fasta")
CLEAN_POS = os.path.join(DATA_DIR, "clean_positives.fasta")
CLEAN_NEG = os.path.join(DATA_DIR, "clean_negatives.fasta")

# ─── ESM-2 ────────────────────────────────────────────
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
# embed_dim for t6_8M=320, t12_35M=480, t30_150M=640
ESM_EMBED_DIM = 480
ESM_BATCH_SIZE = 8

# ─── Data ─────────────────────────────────────────────
MAX_LEN = 100
CD_HIT_THRESHOLD = 0.5
UNIPROT_NEG_LIMIT = 20000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
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
