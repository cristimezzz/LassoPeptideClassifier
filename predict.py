import argparse
import torch
import pandas as pd
from Bio import SeqIO

from config import (
    CHECKPOINT_DIR,
    ESM_MODEL_NAME,
    ESM_BATCH_SIZE,
    PRED_BATCH_SIZE,
    MAX_LEN,
    get_esm_embed_dim,
    get_model_info,
    ensure_dirs,
)
from utils import load_esm_model, extract_esm2_embeddings, load_classifier_from_checkpoint


def predict_fasta(input_fasta, output_csv, checkpoint=None, esm_model_name=None):
    ensure_dirs()
    if esm_model_name is None:
        esm_model_name = ESM_MODEL_NAME

    info = get_model_info(esm_model_name)
    embed_dim = get_esm_embed_dim(esm_model_name)
    batch_size = info["batch"]
    print(f"[*] ESM-2: {info['label']} | embed_dim={embed_dim}")

    print(f"[*] Reading sequences from {input_fasta}")
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if not records:
        print("[-] No sequences found in input file")
        return

    seq_ids = [r.id for r in records]
    print(f"[*] Found {len(records)} sequences, extracting ESM-2 embeddings...")

    esm_model, tokenizer, device = load_esm_model(esm_model_name)
    _, embeddings = extract_esm2_embeddings(
        input_fasta, esm_model_name, esm_model, tokenizer, device, batch_size, MAX_LEN
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
    parser.add_argument("--esm-model", default=None, help="ESM-2 model name")
    args = parser.parse_args()
    predict_fasta(args.input, args.output, args.checkpoint, args.esm_model)


if __name__ == "__main__":
    main()
