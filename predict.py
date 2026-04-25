import argparse
import torch
import pandas as pd
from Bio import SeqIO

from config import (
    CHECKPOINT_DIR,
    ESM_MODEL_NAME,
    MAX_LEN,
    ESM_BATCH_SIZE,
    PRED_BATCH_SIZE,
    get_esm_embed_dim,
    ensure_dirs,
)
from utils import load_esm_model, extract_esm2_embeddings, load_classifier_from_checkpoint


def predict_fasta(input_fasta, output_csv, checkpoint=None):
    ensure_dirs()
    embed_dim = get_esm_embed_dim(ESM_MODEL_NAME)

    print(f"[*] Reading sequences from {input_fasta}")
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if not records:
        print("[-] No sequences found in input file")
        return

    seq_ids = [r.id for r in records]
    print(f"[*] Found {len(records)} sequences, extracting ESM-2 embeddings...")

    esm_model, tokenizer, device = load_esm_model(ESM_MODEL_NAME)
    _, embeddings = extract_esm2_embeddings(
        input_fasta, ESM_MODEL_NAME, esm_model, tokenizer, device, ESM_BATCH_SIZE, MAX_LEN
    )

    ckpt_path = checkpoint or CHECKPOINT_DIR + "/best_model.pt"
    print(f"[*] Loading model from {ckpt_path}")
    model = load_classifier_from_checkpoint(ckpt_path, embed_dim, device)

    with torch.no_grad():
        all_probs = []
        for i in range(0, len(embeddings), PRED_BATCH_SIZE):
            batch = embeddings[i : i + PRED_BATCH_SIZE].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.append(probs.cpu())
        probabilities = torch.cat(all_probs).numpy()

    results = pd.DataFrame({
        "sequence_id": seq_ids,
        "probability": probabilities,
        "prediction": ["positive" if p >= 0.5 else "negative" for p in probabilities],
    })
    results.to_csv(output_csv, index=False)
    print(f"[+] Results saved to {output_csv}")
    print(results.head())


def main():
    parser = argparse.ArgumentParser(description="Predict lasso peptide probability from FASTA")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("-c", "--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()
    predict_fasta(args.input, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
