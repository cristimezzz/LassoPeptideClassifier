import os
import torch
import subprocess
from Bio import SeqIO
from transformers import EsmTokenizer, EsmModel
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
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def run_cd_hit(input_fasta, output_fasta, threshold=0.5):
    print(f"[*] Running CD-HIT (threshold={threshold})...")
    cmd = f"cd-hit -i {input_fasta} -o {output_fasta} -c {threshold} -n 3 -M 0 -d 0"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    print(f"[+] Saved: {output_fasta}")


def extract_esm2_embeddings(fasta_file, model_name=ESM_MODEL_NAME, batch_size=ESM_BATCH_SIZE, max_len=MAX_LEN):
    print(f"[*] Loading ESM-2 model ({model_name})...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    records = list(SeqIO.parse(fasta_file, "fasta"))
    sequences = [str(record.seq) for record in records]
    ids = [record.id for record in records]

    print(f"[*] Extracting embeddings for {len(sequences)} sequences...")
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            last_hidden = last_hidden * mask

            all_embeddings.append(last_hidden.cpu())

    return ids, torch.cat(all_embeddings, dim=0)


def create_and_split_dataset(pos_fasta, neg_fasta, output_dir=DATASET_DIR):
    os.makedirs(output_dir, exist_ok=True)

    pos_ids, pos_features = extract_esm2_embeddings(pos_fasta)
    pos_labels = torch.ones(pos_features.size(0), dtype=torch.float32)

    neg_ids, neg_features = extract_esm2_embeddings(neg_fasta)

    if neg_features.size(0) > pos_features.size(0) * 3:
        idx = torch.randperm(neg_features.size(0))[: pos_features.size(0) * 3]
        neg_features = neg_features[idx]
        neg_ids = [neg_ids[i] for i in idx]

    neg_labels = torch.zeros(neg_features.size(0), dtype=torch.float32)

    X = torch.cat([pos_features, neg_features], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)
    all_ids = pos_ids + neg_ids

    print(f"[*] Feature matrix shape: {X.shape}")
    print(f"[*] Positives: {len(pos_ids)}, Negatives: {len(neg_ids)}")

    test_size_total = TEST_RATIO
    val_size_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_temp, X_test, y_temp, y_test, id_temp, id_test = train_test_split(
        X, y, all_ids, test_size=test_size_total, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=val_size_relative,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    torch.save({"X": X_train, "y": y_train, "ids": id_train}, os.path.join(output_dir, "train.pt"))
    torch.save({"X": X_val, "y": y_val, "ids": id_val}, os.path.join(output_dir, "val.pt"))
    torch.save({"X": X_test, "y": y_test, "ids": id_test}, os.path.join(output_dir, "test.pt"))
    print(f"[+] Dataset saved to {output_dir}/")


if __name__ == "__main__":
    from config import RAW_POS, RAW_NEG

    run_cd_hit(RAW_POS, CLEAN_POS, threshold=CD_HIT_THRESHOLD)
    run_cd_hit(RAW_NEG, CLEAN_NEG, threshold=CD_HIT_THRESHOLD)
    create_and_split_dataset(CLEAN_POS, CLEAN_NEG)
