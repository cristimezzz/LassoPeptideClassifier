import os
import torch
import subprocess
from sklearn.model_selection import train_test_split

from config import (
    CLEAN_POS,
    CLEAN_NEG,
    DATASET_DIR,
    ESM_MODEL_NAME,
    MAX_LEN,
    ESM_BATCH_SIZE,
    CD_HIT_THRESHOLD,
    RANDOM_SEED,
    get_esm_embed_dim,
    ensure_dirs,
)
from utils import extract_esm2_embeddings, load_esm_model


def run_cd_hit(input_fasta, output_fasta, threshold=0.5):
    print(f"[*] Running CD-HIT (threshold={threshold})...")
    try:
        cmd = f"cd-hit -i {input_fasta} -o {output_fasta} -c {threshold} -n 3 -M 0 -d 0"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        n_before = sum(1 for _ in open(input_fasta) if True)
        n_after = sum(1 for _ in open(output_fasta) if True)
        print(f"[+] CD-HIT ok: {n_before//2} → {n_after//2} sequences → {output_fasta}")
    except subprocess.CalledProcessError as e:
        print(f"[!] CD-HIT failed (is it installed?). Skipping deduplication.")
        print(f"    Error: {e}")


def create_and_split_dataset(pos_fasta, neg_fasta, output_dir=DATASET_DIR):
    ensure_dirs()

    # Load ESM-2 once
    esm_model, tokenizer, device = load_esm_model(ESM_MODEL_NAME)
    embed_dim = get_esm_embed_dim(ESM_MODEL_NAME)
    print(f"    ESM-2 embed_dim: {embed_dim}")

    # Downsample negatives *before* extraction to save time
    from Bio import SeqIO
    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))

    if len(neg_records) > len(pos_records) * 3:
        rng = torch.Generator().manual_seed(RANDOM_SEED)
        idx = torch.randperm(len(neg_records), generator=rng)[: len(pos_records) * 3]
        neg_records = [neg_records[i] for i in idx]
        print(f"[i] Downsampled negatives: {len(neg_records)} (x{len(pos_records)*3})")

    # Write downsampled temp files for extract_esm2 (which reads FASTA)
    import tempfile
    tmp_pos = os.path.join(tempfile.gettempdir(), "_pos.fasta")
    tmp_neg = os.path.join(tempfile.gettempdir(), "_neg.fasta")
    from Bio.SeqIO import write
    write(pos_records, tmp_pos, "fasta")
    write(neg_records, tmp_neg, "fasta")

    # Extract embeddings
    pos_ids, pos_features = extract_esm2_embeddings(
        tmp_pos, ESM_MODEL_NAME, esm_model, tokenizer, device, ESM_BATCH_SIZE, MAX_LEN
    )
    pos_labels = torch.ones(pos_features.size(0), dtype=torch.float32)

    neg_ids, neg_features = extract_esm2_embeddings(
        tmp_neg, ESM_MODEL_NAME, esm_model, tokenizer, device, ESM_BATCH_SIZE, MAX_LEN
    )
    neg_labels = torch.zeros(neg_features.size(0), dtype=torch.float32)

    # Clean up temp files
    os.unlink(tmp_pos)
    os.unlink(tmp_neg)

    X = torch.cat([pos_features, neg_features], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)
    all_ids = pos_ids + neg_ids

    print(f"[*] Feature tensor shape: {X.shape}")
    print(f"[*] Pos: {len(pos_ids)}, Neg: {len(neg_ids)}")

    X_temp, X_test, y_temp, y_test, id_temp, id_test = train_test_split(
        X, y, all_ids, test_size=0.1, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.111, random_state=RANDOM_SEED, stratify=y_temp
    )

    for name, x, yy, ids in [
        ("train", X_train, y_train, id_train),
        ("val", X_val, y_val, id_val),
        ("test", X_test, y_test, id_test),
    ]:
        torch.save({"X": x, "y": yy, "ids": ids}, os.path.join(output_dir, f"{name}.pt"))

    print(f"[+] Dataset saved → {output_dir}/")


if __name__ == "__main__":
    from config import RAW_POS, RAW_NEG

    ensure_dirs()
    run_cd_hit(RAW_POS, CLEAN_POS, threshold=CD_HIT_THRESHOLD)
    run_cd_hit(RAW_NEG, CLEAN_NEG, threshold=CD_HIT_THRESHOLD)
    create_and_split_dataset(CLEAN_POS, CLEAN_NEG)
